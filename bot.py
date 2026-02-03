import time
import logging
from datetime import datetime

from window_capture import WindowCapture, ForbiddenAreaOverlay
from image_matcher import ImageMatcher
from mouse_controller import MouseController
from state_machine import StateMachine, State
from telegram_notifier import TelegramNotifier
from asset_scanner import AssetScanner
import config

logger = logging.getLogger(__name__)


class AdaptiveTuner:
    def __init__(self):
        self.enabled = config.ADAPTIVE_TUNER_ENABLED
        self.alpha = config.ADAPTIVE_TUNER_ALPHA
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = config.CLICK_DELAY
        self.move_delay = config.MOUSE_MOVE_DELAY
        self.upgrade_click_interval = config.UPGRADE_CLICK_INTERVAL
        self.search_interval = config.UPGRADE_SEARCH_INTERVAL

    def _ema(self, current, new_value):
        return (1 - self.alpha) * current + self.alpha * new_value

    def record_click_result(self, success):
        if not self.enabled:
            return
        self.click_success_rate = self._ema(self.click_success_rate, 1.0 if success else 0.0)
        self._adjust_click_timing()

    def record_search_result(self, success):
        if not self.enabled:
            return
        self.search_success_rate = self._ema(self.search_success_rate, 1.0 if success else 0.0)
        self._adjust_search_timing()

    def _adjust_click_timing(self):
        if self.click_success_rate < 0.85:
            self.click_delay = min(self.click_delay + 0.01, config.ADAPTIVE_TUNER_MAX_CLICK_DELAY)
            self.move_delay = min(self.move_delay + 0.001, config.ADAPTIVE_TUNER_MAX_MOVE_DELAY)
        elif self.click_success_rate > 0.97:
            self.click_delay = max(self.click_delay - 0.005, config.ADAPTIVE_TUNER_MIN_CLICK_DELAY)
            self.move_delay = max(self.move_delay - 0.001, config.ADAPTIVE_TUNER_MIN_MOVE_DELAY)

    def _adjust_search_timing(self):
        if self.search_success_rate < 0.7:
            self.search_interval = min(self.search_interval + 0.01, config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL)
            self.upgrade_click_interval = min(
                self.upgrade_click_interval + 0.001,
                config.ADAPTIVE_TUNER_MAX_UPGRADE_INTERVAL,
            )
        elif self.search_success_rate > 0.9:
            self.search_interval = max(self.search_interval - 0.005, config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL)
            self.upgrade_click_interval = max(
                self.upgrade_click_interval - 0.001,
                config.ADAPTIVE_TUNER_MIN_UPGRADE_INTERVAL,
            )


class EatventureBot:
    def __init__(self):
        logger.info("Initializing Eatventure Bot...")
        
        self.window_capture = WindowCapture(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.image_matcher = ImageMatcher(config.MATCH_THRESHOLD)
        self.mouse_controller = MouseController(
            self.window_capture.hwnd,
            config.CLICK_DELAY
        )
        self.state_machine = StateMachine(State.FIND_RED_ICONS)
        
        self.register_states()
        self.state_machine.set_priority_resolver(self.resolve_priority_state)
        self.red_icon_templates = [
            "RedIcon", "RedIcon2", "RedIcon3", "RedIcon4", "RedIcon5", "RedIcon6",
            "RedIcon7", "RedIcon8", "RedIcon9", "RedIcon10", "RedIcon11", "RedIcon12",
            "RedIcon13", "RedIcon14", "RedIcon15", "RedIconNoBG"
        ]
        self.templates = self.load_templates()
        self.available_red_icon_templates = self._resolve_red_icon_templates()
        self.running = False
        self.red_icon_cycle_count = 0
        self.red_icons = []
        self.current_red_icon_index = 0
        self.wait_for_unlock_attempts = 0
        self.max_wait_for_unlock_attempts = 4
        
        self.scroll_direction = 'down'
        self.scroll_count = 0
        self.max_scroll_count = 5
        self.work_done = False
        self.cycle_counter = 0
        self.red_icon_processed_count = 0
        self.forbidden_icon_scrolls = 0
        
        self.successful_red_icon_positions = []
        self.upgrade_found_in_cycle = False
        self.consecutive_failed_cycles = 0
        
        self.total_levels_completed = 0
        self.current_level_start_time = None
        
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, config.TELEGRAM_ENABLED)
        self.tuner = AdaptiveTuner()
        self._capture_cache = {}
        self._capture_cache_ttl = config.CAPTURE_CACHE_TTL
        self._new_level_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}

        self.forbidden_zones = [
            (config.FORBIDDEN_ZONE_1_X_MIN, config.FORBIDDEN_ZONE_1_X_MAX,
             config.FORBIDDEN_ZONE_1_Y_MIN, config.FORBIDDEN_ZONE_1_Y_MAX),
            (config.FORBIDDEN_ZONE_2_X_MIN, config.FORBIDDEN_ZONE_2_X_MAX,
             config.FORBIDDEN_ZONE_2_Y_MIN, config.FORBIDDEN_ZONE_2_Y_MAX),
            (config.FORBIDDEN_ZONE_3_X_MIN, config.FORBIDDEN_ZONE_3_X_MAX,
             config.FORBIDDEN_ZONE_3_Y_MIN, config.FORBIDDEN_ZONE_3_Y_MAX),
            (config.FORBIDDEN_ZONE_4_X_MIN, config.FORBIDDEN_ZONE_4_X_MAX,
             config.FORBIDDEN_ZONE_4_Y_MIN, config.FORBIDDEN_ZONE_4_Y_MAX),
            (config.FORBIDDEN_ZONE_5_X_MIN, config.FORBIDDEN_ZONE_5_X_MAX,
             config.FORBIDDEN_ZONE_5_Y_MIN, config.FORBIDDEN_ZONE_5_Y_MAX),
            (config.FORBIDDEN_ZONE_6_X_MIN, config.FORBIDDEN_ZONE_6_X_MAX,
             config.FORBIDDEN_ZONE_6_Y_MIN, config.FORBIDDEN_ZONE_6_Y_MAX),
        ]

        self.overlay = None
        if config.ShowForbiddenArea:
            self.overlay = ForbiddenAreaOverlay(self.window_capture.hwnd, self.forbidden_zones)
            self.overlay.start()
            logger.info("Forbidden area overlay enabled and started")
        
        logger.info("Bot initialized successfully")

    def _apply_tuning(self):
        if not self.tuner.enabled:
            return
        self.mouse_controller.click_delay = self.tuner.click_delay
        config.MOUSE_MOVE_DELAY = self.tuner.move_delay

    def _scroll_away_from_forbidden_zone(self, y_position):
        if self.forbidden_icon_scrolls >= config.FORBIDDEN_ICON_MAX_SCROLLS:
            logger.warning("Max forbidden-icon scrolls reached; skipping icon")
            return False

        if y_position >= config.FORBIDDEN_ZONE_5_Y_MIN or y_position >= config.FORBIDDEN_CLICK_Y_MIN:
            direction = "up"
        elif y_position <= config.FORBIDDEN_ZONE_6_Y_MAX or y_position <= config.FORBIDDEN_ZONE_4_Y_MAX:
            direction = "down"
        else:
            direction = self.scroll_direction

        if direction == "up":
            logger.info("Red icon in forbidden zone → scrolling up to clear")
            self.mouse_controller.drag(
                config.SCROLL_END_POS[0], config.SCROLL_END_POS[1],
                config.SCROLL_START_POS[0], config.SCROLL_START_POS[1],
                duration=config.FORBIDDEN_ICON_SCROLL_DURATION,
                relative=True,
            )
        else:
            logger.info("Red icon in forbidden zone → scrolling down to clear")
            self.mouse_controller.drag(
                config.SCROLL_START_POS[0], config.SCROLL_START_POS[1],
                config.SCROLL_END_POS[0], config.SCROLL_END_POS[1],
                duration=config.FORBIDDEN_ICON_SCROLL_DURATION,
                relative=True,
            )

        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        if config.FORBIDDEN_ICON_SCROLL_COOLDOWN > 0:
            time.sleep(config.FORBIDDEN_ICON_SCROLL_COOLDOWN)
        self.forbidden_icon_scrolls += 1
        return True

    def resolve_priority_state(self, current_state):
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
        found, confidence, x, y = self._detect_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        )
        if found:
            logger.info("Priority override: new level detected, transitioning immediately")
            return State.TRANSITION_LEVEL

        return None

    def _capture(self, max_y=None, force=False):
        cache_key = max_y if max_y is not None else "full"
        cached = self._capture_cache.get(cache_key)
        now = time.monotonic()
        if not force and cached and now - cached[0] <= self._capture_cache_ttl:
            return cached[1]

        frame = self.window_capture.capture(max_y=max_y)
        self._capture_cache[cache_key] = (now, frame)
        return frame

    def _clear_capture_cache(self):
        self._capture_cache.clear()
        self._new_level_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}

    def _sleep_until(self, target_time):
        now = time.monotonic()
        if target_time > now:
            time.sleep(target_time - now)

    def _detect_new_level(self, screenshot=None, max_y=None, force=False):
        target_max_y = max_y if max_y is not None else config.MAX_SEARCH_Y
        now = time.monotonic()
        cached = self._new_level_cache
        if not force and cached["max_y"] == target_max_y and now - cached["timestamp"] <= self._capture_cache_ttl:
            return cached["result"]

        if screenshot is None:
            screenshot = self._capture(max_y=target_max_y, force=force)

        result = self._find_new_level(screenshot, threshold=config.NEW_LEVEL_THRESHOLD)
        self._new_level_cache = {"timestamp": now, "result": result, "max_y": target_max_y}
        return result

    def _should_interrupt_for_new_level(self, screenshot=None, max_y=None, force=False):
        found, confidence, x, y = self._detect_new_level(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
        )
        if found:
            logger.info("Priority override: new level detected, interrupting current action")
        return found

    def _find_new_level(self, screenshot, threshold=None):
        if "newLevel" not in self.templates:
            return False, 0.0, 0, 0

        template, mask = self.templates["newLevel"]
        return self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=threshold or config.NEW_LEVEL_THRESHOLD,
            template_name="newLevel",
        )

    def _resolve_red_icon_templates(self):
        resolved = []
        for template_name in self.red_icon_templates:
            if template_name not in self.templates:
                continue
            template, mask = self.templates[template_name]
            resolved.append((template_name, template, mask))
        return resolved

    def _has_stats_upgrade_icon(self, screenshot):
        if not self.available_red_icon_templates:
            return False

        height, width = screenshot.shape[:2]
        x_min = max(0, config.UPGRADE_RED_ICON_X_MIN - config.STATS_ICON_PADDING)
        x_max = min(width, config.UPGRADE_RED_ICON_X_MAX + config.STATS_ICON_PADDING)
        y_min = max(0, config.UPGRADE_RED_ICON_Y_MIN - config.STATS_ICON_PADDING)
        y_max = min(height, config.UPGRADE_RED_ICON_Y_MAX + config.STATS_ICON_PADDING)

        if x_min >= x_max or y_min >= y_max:
            return False

        roi = screenshot[y_min:y_max, x_min:x_max]

        for template_name, template, mask in self.available_red_icon_templates:
            icons = self.image_matcher.find_all_templates(
                roi,
                template,
                mask=mask,
                threshold=config.STATS_RED_ICON_THRESHOLD,
                min_distance=80,
                template_name=template_name,
            )

            if icons:
                return True

        return False

    def _merge_detection(self, detections, buckets, x, y, template_name, conf, proximity=10, bucket_size=10):
        bucket_x = x // bucket_size
        bucket_y = y // bucket_size
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in buckets.get((bucket_x + dx, bucket_y + dy), []):
                    if abs(x - px) < proximity and abs(y - py) < proximity:
                        detections[(px, py)].append((template_name, conf))
                        return

        detections[(x, y)] = [(template_name, conf)]
        buckets.setdefault((bucket_x, bucket_y), []).append((x, y))
    
    def load_templates(self):
        required_templates = self._required_template_names()
        scanner = AssetScanner(self.image_matcher)
        return scanner.scan(config.ASSETS_DIR, required_templates=required_templates)

    def _required_template_names(self):
        box_names = [f"box{i}" for i in range(1, 6)]
        required = set(self.red_icon_templates)
        required.update(["newLevel", "unlock", "upgradeStation"])
        required.update(box_names)
        return required
    
    def register_states(self):
        self.state_machine.register_handler(State.FIND_RED_ICONS, self.handle_find_red_icons)
        self.state_machine.register_handler(State.CLICK_RED_ICON, self.handle_click_red_icon)
        self.state_machine.register_handler(State.CHECK_UNLOCK, self.handle_check_unlock)
        self.state_machine.register_handler(State.SEARCH_UPGRADE_STATION, self.handle_search_upgrade_station)
        self.state_machine.register_handler(State.HOLD_UPGRADE_STATION, self.handle_hold_upgrade_station)
        self.state_machine.register_handler(State.OPEN_BOXES, self.handle_open_boxes)
        self.state_machine.register_handler(State.UPGRADE_STATS, self.handle_upgrade_stats)
        self.state_machine.register_handler(State.SCROLL, self.handle_scroll)
        self.state_machine.register_handler(State.CHECK_NEW_LEVEL, self.handle_check_new_level)
        self.state_machine.register_handler(State.TRANSITION_LEVEL, self.handle_transition_level)
        self.state_machine.register_handler(State.WAIT_FOR_UNLOCK, self.handle_wait_for_unlock)
    
    def handle_find_red_icons(self, current_state):
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)

        self.work_done = False
        self.forbidden_icon_scrolls = 0
        
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]

        found, confidence, x, y = self._detect_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        )
        if found:
            logger.info(f"newLevel.png found at ({x}, {y}), transitioning to new level")
            return State.TRANSITION_LEVEL
        
        all_detections = {}
        all_detections_extended = {}
        detection_buckets = {}
        detection_buckets_extended = {}
        
        for template_name, template, mask in self.available_red_icon_templates:
            icons = self.image_matcher.find_all_templates(
                screenshot, template, mask=mask,
                threshold=config.RED_ICON_THRESHOLD,
                min_distance=80,
                template_name=template_name,
            )
            
            for conf, x, y in icons:
                self._merge_detection(
                    all_detections_extended,
                    detection_buckets_extended,
                    x,
                    y,
                    template_name,
                    conf,
                )

                if y <= config.MAX_SEARCH_Y:
                    self._merge_detection(
                        all_detections,
                        detection_buckets,
                        x,
                        y,
                        template_name,
                        conf,
                    )
        
        min_matches = config.RED_ICON_MIN_MATCHES
        total_detections = len(all_detections)
        rejected_count = 0
        
        self.red_icons = []
        for (x, y), matches in all_detections.items():
            if len(matches) >= min_matches:
                max_conf = max(conf for _, conf in matches)
                self.red_icons.append((max_conf, x, y))
            else:
                rejected_count += 1
        
        all_red_icons_extended = []
        for (x, y), matches in all_detections_extended.items():
            if len(matches) >= min_matches:
                max_conf = max(conf for _, conf in matches)
                all_red_icons_extended.append((max_conf, x, y))
        
        logger.info(f"Red Icon Detection: {total_detections} total → {len(self.red_icons)} valid (min {min_matches} template matches), {rejected_count} rejected")
        
        for conf, x, y in all_red_icons_extended:
            if (config.NEW_LEVEL_RED_ICON_X_MIN <= x <= config.NEW_LEVEL_RED_ICON_X_MAX and 
                config.NEW_LEVEL_RED_ICON_Y_MIN <= y <= config.NEW_LEVEL_RED_ICON_Y_MAX):
                logger.info(f"New level detected! Red icon at ({x}, {y})")
                return State.CHECK_NEW_LEVEL
        
        if not self.red_icons:
            logger.info("No valid red icons after scan; scrolling to search")
            return State.SCROLL
        else:
            filtered_icons = []
            forbidden_zone_count = 0
            forbidden_zones = self.forbidden_zones

            for conf, x, y in self.red_icons:
                in_forbidden = any(
                    zone_x_min <= x <= zone_x_max and zone_y_min <= y <= zone_y_max
                    for zone_x_min, zone_x_max, zone_y_min, zone_y_max in forbidden_zones
                )
                
                if in_forbidden:
                    forbidden_zone_count += 1
                else:
                    filtered_icons.append((conf, x, y))
            
            if forbidden_zone_count > 0:
                logger.info(f"Forbidden Zone Filter: {forbidden_zone_count} icons removed")
            
            if not filtered_icons:
                logger.info("No valid red icons after filtering; scrolling to search")
                return State.SCROLL
            
            def get_priority(icon):
                conf, x, y = icon
                for success_y in self.successful_red_icon_positions:
                    if abs(y - success_y) < 50:
                        return (0, y)
                return (1, y)
            
            filtered_icons.sort(key=get_priority)
            self.red_icons = filtered_icons
            
            logger.info(f"✓ {len(self.red_icons)} red icons ready to process")
            self.current_red_icon_index = 0
            self.red_icon_cycle_count = 0
            self.work_done = True
            return State.CLICK_RED_ICON
    
    def handle_click_red_icon(self, current_state):
        if self.current_red_icon_index >= len(self.red_icons):
            logger.info("All red icons processed, continuing cycle")
            return State.OPEN_BOXES
        
        confidence, x, y = self.red_icons[self.current_red_icon_index]
        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y
        
        if self.mouse_controller.is_in_forbidden_zone(click_x, click_y):
            logger.warning(f"Red icon click blocked - position with offset ({click_x}, {click_y}) is in forbidden zone")
            if self._scroll_away_from_forbidden_zone(click_y):
                return State.FIND_RED_ICONS
            self.current_red_icon_index += 1
            return State.CLICK_RED_ICON if self.current_red_icon_index < len(self.red_icons) else State.OPEN_BOXES
        
        logger.info(f"Clicking red icon {self.current_red_icon_index + 1}/{len(self.red_icons)} at ({click_x}, {click_y})")
        click_success = self.mouse_controller.click(click_x, click_y, relative=True)
        self.tuner.record_click_result(click_success)
        self._apply_tuning()
        
        self.red_icon_cycle_count = 0
        return State.CHECK_UNLOCK
    
    def handle_check_unlock(self, current_state):
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
        
        if "unlock" in self.templates:
            template, mask = self.templates["unlock"]
            found, confidence, x, y = self.image_matcher.find_template(
                limited_screenshot, template, mask=mask,
                threshold=config.UNLOCK_THRESHOLD, template_name="unlock"
            )
            
            if found:
                if self.mouse_controller.is_in_forbidden_zone(x, y):
                    logger.warning(f"Unlock button in forbidden zone, skipping")
                else:
                    logger.info(f"Unlock found, clicking")
                    self.mouse_controller.click(x, y, relative=True)
        
        return State.SEARCH_UPGRADE_STATION
    
    def handle_search_upgrade_station(self, current_state):
        max_attempts = 5
        relaxed_threshold = config.UPGRADE_STATION_THRESHOLD - 0.05
        retry_delay = self.tuner.search_interval
        
        for attempt in range(max_attempts):
            limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
            
            if "upgradeStation" in self.templates:
                template, mask = self.templates["upgradeStation"]
                
                current_threshold = config.UPGRADE_STATION_THRESHOLD if attempt < 2 else relaxed_threshold
                
                found, confidence, x, y = self.image_matcher.find_template(
                    limited_screenshot, template, mask=mask,
                    threshold=current_threshold, template_name="upgradeStation"
                )
                
                if found:
                    logger.info(f"✓ Upgrade station found (attempt {attempt + 1})")
                    self.upgrade_station_pos = (x, y)
                    
                    if self.current_red_icon_index < len(self.red_icons):
                        _, _, red_y = self.red_icons[self.current_red_icon_index]
                        if red_y not in self.successful_red_icon_positions:
                            self.successful_red_icon_positions.append(red_y)
                    
                    self.upgrade_found_in_cycle = True
                    self.consecutive_failed_cycles = 0
                    self.tuner.record_search_result(True)
                    self._apply_tuning()
                    return State.HOLD_UPGRADE_STATION
            
            if attempt < max_attempts - 1:
                if retry_delay > 0:
                    time.sleep(retry_delay)
        
        logger.info(f"✗ Upgrade station not found (failed cycles: {self.consecutive_failed_cycles + 1})")
        self.tuner.record_search_result(False)
        self._apply_tuning()
        self.red_icon_processed_count += 1
        self.consecutive_failed_cycles += 1
        return State.OPEN_BOXES
    
    def handle_hold_upgrade_station(self, current_state):
        x, y = self.upgrade_station_pos

        if self.mouse_controller.is_in_forbidden_zone(x, y):
            logger.warning("Upgrade station position is in forbidden zone; skipping clicks")
            self.red_icon_processed_count += 1
            return State.OPEN_BOXES
        
        logger.info("Spamming upgrade station clicks...")
        
        max_hold_time = config.UPGRADE_HOLD_DURATION
        click_interval = self.tuner.upgrade_click_interval
        check_interval = config.UPGRADE_CHECK_INTERVAL
        start_time = time.monotonic()
        upgrade_missing_logged = False
        next_click_time = start_time
        next_check_time = start_time + check_interval
        
        while True:
            now = time.monotonic()
            if now - start_time >= max_hold_time:
                break
            
            if now >= next_click_time:
                self.mouse_controller.click(x, y, relative=True, wait_after=False)
                next_click_time = max(next_click_time + click_interval, now + click_interval)
            
            if now >= next_check_time:
                limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)

                if "upgradeStation" in self.templates:
                    template, mask = self.templates["upgradeStation"]
                    found, confidence, found_x, found_y = self.image_matcher.find_template(
                        limited_screenshot, template, mask=mask,
                        threshold=config.UPGRADE_STATION_THRESHOLD, template_name="upgradeStation",
                        check_color=config.UPGRADE_STATION_COLOR_CHECK
                    )
                    
                    if not found and not upgrade_missing_logged:
                        logger.info("Upgrade station not found while clicking; continuing until duration completes.")
                        upgrade_missing_logged = True

                if self._should_interrupt_for_new_level(
                    screenshot=limited_screenshot,
                    max_y=config.MAX_SEARCH_Y,
                    force=True,
                ):
                    return State.TRANSITION_LEVEL

                next_check_time = max(next_check_time + check_interval, now + check_interval)

            now = time.monotonic()
            next_action_time = min(next_click_time, next_check_time, start_time + max_hold_time)
            self._sleep_until(next_action_time)
        
        elapsed_time = time.monotonic() - start_time
        logger.info(f"Clicking complete: max time reached ({elapsed_time:.1f}s)")
        
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            time.sleep(config.IDLE_CLICK_SETTLE_DELAY)
        
        self.red_icon_processed_count += 1
        
        logger.info("✓ Upgrade station complete → Stats upgrade next")
        return State.UPGRADE_STATS
    
    def handle_upgrade_stats(self, current_state):
        logger.info("⬆ Stats upgrade starting")
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        
        extended_screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = extended_screenshot[:config.MAX_SEARCH_Y, :]

        found, confidence, x, y = self._detect_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        )
        if found:
            logger.info("New level detected during stats upgrade")
            return State.TRANSITION_LEVEL
        
        if not self._has_stats_upgrade_icon(extended_screenshot):
            logger.info("✗ No stats icon, skipping")
            return State.SCROLL
        
        logger.info("✓ Stats icon found, upgrading")
        self.mouse_controller.click(config.STATS_UPGRADE_BUTTON_POS[0], config.STATS_UPGRADE_BUTTON_POS[1], relative=True)
        time.sleep(config.STATE_DELAY)
        
        start_time = time.monotonic()
        last_new_level_check = 0.0
        next_click_time = start_time
        while time.monotonic() - start_time < config.STATS_UPGRADE_CLICK_DURATION:
            self._sleep_until(next_click_time)
            self.mouse_controller.click(
                config.STATS_UPGRADE_POS[0],
                config.STATS_UPGRADE_POS[1],
                relative=True,
                wait_after=False,
            )
            next_click_time = max(
                next_click_time + config.STATS_UPGRADE_CLICK_DELAY,
                time.monotonic() + config.STATS_UPGRADE_CLICK_DELAY,
            )
            elapsed = time.monotonic() - start_time
            if elapsed - last_new_level_check >= config.NEW_LEVEL_INTERRUPT_INTERVAL:
                limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
                if self._should_interrupt_for_new_level(
                    screenshot=limited_screenshot,
                    max_y=config.MAX_SEARCH_Y,
                ):
                    return State.TRANSITION_LEVEL
                last_new_level_check = elapsed
        
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        logger.info("========== STAT UPGRADE COMPLETED ==========")
        return State.OPEN_BOXES
    
    def handle_open_boxes(self, current_state):
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

        found, confidence, x, y = self._detect_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        )
        if found:
            logger.info("New level found, transitioning")
            return State.TRANSITION_LEVEL
        
        box_names = ["box1", "box2", "box3", "box4", "box5"]
        boxes_found = 0
        
        for box_name in box_names:
            if box_name in self.templates:
                template, mask = self.templates[box_name]
                found, confidence, x, y = self.image_matcher.find_template(
                    limited_screenshot, template, mask=mask,
                    threshold=config.BOX_THRESHOLD, template_name=box_name
                )
                
                if found:
                    if self.mouse_controller.is_in_forbidden_zone(x, y):
                        logger.debug(f"{box_name} in forbidden zone, skipping")
                    else:
                        self.mouse_controller.click(x, y, relative=True)
                        boxes_found += 1
        
        if self._should_interrupt_for_new_level(
            max_y=config.MAX_SEARCH_Y,
            force=True,
        ):
            logger.info("New level detected while opening boxes")
            return State.TRANSITION_LEVEL
        
        if boxes_found > 0:
            logger.info(f"🎁 Opened {boxes_found} boxes")
            self.work_done = True
        
        if self.upgrade_found_in_cycle:
            logger.info("✓ Upgrade found → Staying in area")
            self.upgrade_found_in_cycle = False
            self.cycle_counter = 0
            return State.FIND_RED_ICONS
        
        self.cycle_counter += 1
        
        if self.consecutive_failed_cycles >= 3:
            logger.info(f"⚠ {self.consecutive_failed_cycles} failed → Force scroll")
            self.consecutive_failed_cycles = 0
            self.cycle_counter = 0
            return State.SCROLL
        
        if self.cycle_counter >= 2:
            logger.info(f"➡ Cycle {self.cycle_counter}/2 done → Scrolling")
            self.cycle_counter = 0
            return State.SCROLL
        else:
            return State.FIND_RED_ICONS
    
    def handle_scroll(self, current_state):
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

        found, confidence, x, y = self._detect_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        )
        if found:
            logger.info("New level detected before scroll")
            return State.TRANSITION_LEVEL
        
        if self.scroll_direction == 'up':
            logger.info(f"⬆ Scroll UP ({self.scroll_count + 1}/{self.max_scroll_count})")
            self.mouse_controller.drag(
                config.SCROLL_END_POS[0], config.SCROLL_END_POS[1],
                config.SCROLL_START_POS[0], config.SCROLL_START_POS[1],
                duration=config.SCROLL_DURATION, relative=True
            )
        else:  # down
            logger.info(f"⬇ Scroll DOWN ({self.scroll_count + 1}/{self.max_scroll_count})")
            self.mouse_controller.drag(
                config.SCROLL_START_POS[0], config.SCROLL_START_POS[1],
                config.SCROLL_END_POS[0], config.SCROLL_END_POS[1],
                duration=config.SCROLL_DURATION, relative=True
            )
        
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        
        self.scroll_count += 1
        
        if self.scroll_count >= self.max_scroll_count:
            if self.scroll_direction == 'up':
                self.scroll_direction = 'down'
            else:
                self.scroll_direction = 'up'
            self.scroll_count = 0
        
        return State.FIND_RED_ICONS
    
    def handle_check_new_level(self, current_state):
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            time.sleep(config.IDLE_CLICK_SETTLE_DELAY)

        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
        if self._should_interrupt_for_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        ):
            return State.TRANSITION_LEVEL
        
        logger.info("Clicking new level button position")
        self.mouse_controller.click(config.NEW_LEVEL_BUTTON_POS[0], config.NEW_LEVEL_BUTTON_POS[1], relative=True)
        if config.NEW_LEVEL_BUTTON_DELAY > 0:
            time.sleep(config.NEW_LEVEL_BUTTON_DELAY)
        
        logger.info("Triggering follow-up click after new level check")
        self.mouse_controller.click(166, 526, relative=True)
        if config.NEW_LEVEL_FOLLOWUP_DELAY > 0:
            time.sleep(config.NEW_LEVEL_FOLLOWUP_DELAY)

        self.scroll_direction = 'down'
        self.scroll_count = 0
        return State.FIND_RED_ICONS
    
    def handle_transition_level(self, current_state):
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        
        max_attempts = 5
        
        for attempt in range(max_attempts):
            limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

            found, confidence, x, y = self._detect_new_level(
                screenshot=limited_screenshot,
                max_y=config.MAX_SEARCH_Y,
            )
            if found:
                logger.info(f"New level button found at ({x}, {y}) (attempt {attempt + 1})")
                self.mouse_controller.click(x, y, relative=True)
                if config.TRANSITION_POST_CLICK_DELAY > 0:
                    time.sleep(config.TRANSITION_POST_CLICK_DELAY)

                self.total_levels_completed += 1

                time_spent = 0
                if self.current_level_start_time:
                    time_spent = (datetime.now() - self.current_level_start_time).total_seconds()

                self.current_level_start_time = datetime.now()

                self.telegram.notify_new_level(self.total_levels_completed, time_spent)

                logger.info(f"Level {self.total_levels_completed} completed. Time spent: {time_spent:.1f}s")
                logger.info("Waiting for unlock button after level transition")
                return State.WAIT_FOR_UNLOCK
            
            if attempt < max_attempts - 1:
                if config.TRANSITION_RETRY_DELAY > 0:
                    time.sleep(config.TRANSITION_RETRY_DELAY)
        
        logger.warning("New level button not found after 5 attempts")
        self.scroll_direction = 'down'
        self.scroll_count = 0
        return State.FIND_RED_ICONS
    
    def handle_wait_for_unlock(self, current_state):
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            time.sleep(config.IDLE_CLICK_SETTLE_DELAY)
        
        self.wait_for_unlock_attempts += 1
        logger.debug(f"Waiting for unlock button (attempt {self.wait_for_unlock_attempts}/{self.max_wait_for_unlock_attempts})")
        
        if self.wait_for_unlock_attempts > self.max_wait_for_unlock_attempts:
            logger.warning(f"Unlock button not found after {self.max_wait_for_unlock_attempts} attempts, resetting to scroll")
            self.wait_for_unlock_attempts = 0
            self.scroll_direction = 'down'
            self.scroll_count = 0
            return State.FIND_RED_ICONS
        
        screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

        if "unlock" in self.templates:
            template, mask = self.templates["unlock"]
            found, confidence, x, y = self.image_matcher.find_template(
                screenshot, template, mask=mask,
                threshold=config.UNLOCK_THRESHOLD, template_name="unlock"
            )

            if found:
                logger.info(f"Unlock button found at ({x}, {y}) after level transition")
                self.mouse_controller.click(x, y, relative=True)
                if config.UNLOCK_POST_CLICK_DELAY > 0:
                    time.sleep(config.UNLOCK_POST_CLICK_DELAY)
                logger.info("Starting new level")
                self.wait_for_unlock_attempts = 0
                self.scroll_direction = 'down'
                self.scroll_count = 0
                return State.FIND_RED_ICONS
        
        if config.WAIT_UNLOCK_RETRY_DELAY > 0:
            time.sleep(config.WAIT_UNLOCK_RETRY_DELAY)
        return State.WAIT_FOR_UNLOCK
    
    def run(self):
        self.running = True
        logger.info("Bot started - Press Ctrl+C to stop")
        
        try:
            while self.running:
                if not self.window_capture.is_window_active():
                    logger.error(f"Window '{config.WINDOW_TITLE}' is no longer active!")
                    break
                
                self.step()
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self.stop()

    def step(self):
        self._clear_capture_cache()
        self._apply_tuning()
        self.state_machine.update()
    
    def stop(self):
        self.running = False
        if self.overlay:
            self.overlay.stop()
        logger.info("Bot stopped")
