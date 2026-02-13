import json
import os
import time
import logging
import re
import threading
from datetime import datetime
import numpy as np

from window_capture import WindowCapture, ForbiddenAreaOverlay
from image_matcher import ImageMatcher
from mouse_controller import MouseController
from state_machine import StateMachine, State
from telegram_notifier import TelegramNotifier
from asset_scanner import AssetScanner
from ai_optimizer import AdaptiveTuner, VisionOptimizer, VisionPersistence, HistoricalLearner
import config

logger = logging.getLogger(__name__)


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
        self.available_red_icon_templates = self._build_available_red_icon_templates()
        self._red_template_hit_counts = {}
        self._red_template_priority = []
        self._red_template_last_seen = {}
        self._red_template_decay_window = max(1.0, float(getattr(config, "RED_ICON_STABILITY_CACHE_TTL", 0.22)))
        self.running = False
        self.red_icon_cycle_count = 0
        self.red_icons = []
        self.current_red_icon_index = 0
        self.wait_for_unlock_attempts = 0
        self.max_wait_for_unlock_attempts = 4
        
        self.scroll_direction = 'up'
        self.scroll_count = 0
        self.max_scroll_count = max(1, int(getattr(config, "MAX_SCROLL_CYCLES", 5)))
        self.work_done = False
        self.cycle_counter = 0
        self.red_icon_processed_count = 0
        self.forbidden_icon_scrolls = 0
        
        self.successful_red_icon_positions = []
        self.upgrade_found_in_cycle = False
        self.consecutive_failed_cycles = 0
        self.no_red_icons_found = False
        
        self.total_levels_completed = 0
        self.current_level_start_time = None
        self.completion_detected_time = None
        self.completion_detected_by = None
        
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, config.TELEGRAM_ENABLED)
        self.tuner = AdaptiveTuner()
        self.vision_persistence = VisionPersistence(
            config.AI_VISION_STATE_FILE,
            config.AI_VISION_SAVE_INTERVAL,
        )
        self.vision_optimizer = VisionOptimizer(self.vision_persistence)
        self.vision_optimizer.apply_persisted_state(self.vision_persistence.load())
        self.learning_persistence = VisionPersistence(
            config.AI_LEARNING_STATE_FILE,
            config.AI_LEARNING_SAVE_INTERVAL,
        )
        self.historical_learner = HistoricalLearner(self, self.learning_persistence)
        self._capture_cache = {}
        self._capture_cache_ttl = config.CAPTURE_CACHE_TTL
        self._new_level_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}
        self._new_level_red_icon_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}
        self._capture_lock = threading.Lock()
        self._interrupt_lock = threading.Lock()
        self._new_level_event = threading.Event()
        self._new_level_interrupt = None
        self._new_level_monitor_stop = threading.Event()
        self._new_level_monitor_thread = None
        self._last_upgrade_station_pos = None
        self._last_new_level_override_time = 0.0
        self._last_idle_click_time = 0.0
        self._state_last_run_at = {}
        self._recent_red_icon_history = []
        self._no_red_scroll_cycle_pending = False

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

    def _record_new_level_interrupt(self, source, confidence, x, y):
        with self._interrupt_lock:
            self._new_level_interrupt = {
                "source": source,
                "confidence": confidence,
                "x": x,
                "y": y,
                "timestamp": time.monotonic(),
            }
        self._new_level_event.set()
        self._mark_restaurant_completed(source, confidence)

    def _consume_new_level_interrupt(self):
        with self._interrupt_lock:
            if not self._new_level_event.is_set():
                return None
            interrupt = self._new_level_interrupt
            self._new_level_interrupt = None
            self._new_level_event.clear()
            return interrupt

    def _monitor_new_level(self):
        interval = config.NEW_LEVEL_MONITOR_INTERVAL
        while not self._new_level_monitor_stop.is_set():
            if self._new_level_event.is_set():
                time.sleep(max(interval, 0.01))
                continue

            limited_screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)

            red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
                screenshot=limited_screenshot,
                max_y=config.EXTENDED_SEARCH_Y,
                record_miss=False,
            )
            if red_found:
                logger.info(
                    "Background monitor: new level red icon detected at (%s, %s)",
                    red_x,
                    red_y,
                )
                self._record_new_level_interrupt("new level red icon", red_conf, red_x, red_y)
                time.sleep(max(interval, 0.01))
                continue

            found, confidence, x, y = self._detect_new_level(
                screenshot=limited_screenshot,
                max_y=config.EXTENDED_SEARCH_Y,
                record_miss=False,
            )
            if found:
                logger.info("Background monitor: new level button detected at (%s, %s)", x, y)
                self._record_new_level_interrupt("new level button", confidence, x, y)

            time.sleep(max(interval, 0.01))

    def _apply_tuning(self):
        if not self.tuner.enabled:
            return
        self.mouse_controller.click_delay = self.tuner.click_delay
        config.MOUSE_MOVE_DELAY = self.tuner.move_delay

    def _click_idle(self, wait_after=True):
        now = time.monotonic()
        cooldown = getattr(config, "IDLE_CLICK_COOLDOWN", 0.0)
        if cooldown > 0 and now - self._last_idle_click_time < cooldown:
            logger.debug("Skipping idle click due to cooldown")
            return False
        clicked = self.mouse_controller.click(
            config.IDLE_CLICK_POS[0],
            config.IDLE_CLICK_POS[1],
            relative=True,
            wait_after=wait_after,
        )
        if clicked:
            self._last_idle_click_time = time.monotonic()
        return clicked

    def resolve_priority_state(self, current_state):
        if current_state == State.FIND_RED_ICONS and self._no_red_scroll_cycle_pending:
            logger.info("Priority override: continuing no-red scroll cycle after fallback asset scan")
            self._no_red_scroll_cycle_pending = False
            self.no_red_icons_found = True
            return State.SCROLL

        interrupt = self._consume_new_level_interrupt()
        if interrupt:
            logger.info(
                "Priority override: background %s detected at (%s, %s), interrupting current action",
                interrupt["source"],
                interrupt["x"],
                interrupt["y"],
            )
            if self._no_red_scroll_cycle_pending:
                logger.info("Clearing deferred no-red scroll due to pending level transition interrupt")
                self._no_red_scroll_cycle_pending = False
            
            # Record completion stats before clicking
            self._mark_restaurant_completed(interrupt["source"], interrupt["confidence"])
            
            if self._click_new_level_override(source=interrupt["source"]):
                # Full transition sequence handled (New Level button + Fly button)
                self._finalize_level_completion()
                return State.WAIT_FOR_UNLOCK
            
            return State.TRANSITION_LEVEL

        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        priority_hit = self._detect_new_level_priority(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
            force=True,
            record_miss=False,
        )
        if priority_hit:
            source, confidence, x, y = priority_hit
            logger.info(
                "Priority override: %s detected at (%s, %s), transitioning immediately",
                source,
                x,
                y,
            )
            if self._no_red_scroll_cycle_pending:
                logger.info("Clearing deferred no-red scroll due to immediate level transition")
                self._no_red_scroll_cycle_pending = False
            
            # Record completion stats before clicking
            self._mark_restaurant_completed(source, confidence)
            
            if self._click_new_level_override(source=source):
                # Full transition sequence handled
                self._finalize_level_completion()
                return State.WAIT_FOR_UNLOCK
                
            return State.TRANSITION_LEVEL

        return None

    def _enforce_state_min_interval(self):
        state = self.state_machine.get_state_name()
        per_state = getattr(config, "STATE_MIN_INTERVALS", {})
        min_interval = float(per_state.get(state, getattr(config, "STATE_MIN_INTERVAL_DEFAULT", 0.0)))
        if min_interval <= 0:
            self._state_last_run_at[state] = time.monotonic()
            return

        now = time.monotonic()
        last_run = self._state_last_run_at.get(state, 0.0)
        wait_time = (last_run + min_interval) - now
        if wait_time > 0 and self._sleep_with_interrupt(wait_time):
            self._state_last_run_at[state] = time.monotonic()
            return
        self._state_last_run_at[state] = time.monotonic()

    def _stable_red_icons(self, red_icons):
        if not red_icons:
            return []

        ttl = max(0.01, float(getattr(config, "RED_ICON_STABILITY_CACHE_TTL", 0.22)))
        radius = max(4, int(getattr(config, "RED_ICON_STABILITY_RADIUS", 14)))
        min_hits = max(1, int(getattr(config, "RED_ICON_STABILITY_MIN_HITS", 2)))
        max_history = max(2, int(getattr(config, "RED_ICON_STABILITY_MAX_HISTORY", 10)))
        now = time.monotonic()

        history = []
        for entry in getattr(self, "_recent_red_icon_history", []):
            if now - entry.get("timestamp", 0.0) <= ttl:
                history.append(entry)

        current = {"timestamp": now, "icons": list(red_icons)}
        history.append(current)
        if len(history) > max_history:
            history = history[-max_history:]
        self._recent_red_icon_history = history

        stable = []
        for conf, x, y in red_icons:
            hits = 0
            best_conf = conf
            for entry in history:
                for h_conf, hx, hy in entry["icons"]:
                    if abs(hx - x) <= radius and abs(hy - y) <= radius:
                        hits += 1
                        if h_conf > best_conf:
                            best_conf = h_conf
                        break
            if hits >= min_hits:
                stable.append((best_conf, x, y))

        return stable or red_icons

    def _click_new_level_override(self, source=None):
        now = time.monotonic()
        cooldown = getattr(config, "NEW_LEVEL_OVERRIDE_COOLDOWN", 0.0)
        if cooldown > 0 and now - self._last_new_level_override_time < cooldown:
            logger.debug("Priority override: skipping click sequence due to cooldown")
            return False
        self._last_new_level_override_time = now

        logger.info(
            "Priority override: clicking new level position at (%s, %s)",
            config.NEW_LEVEL_POS[0],
            config.NEW_LEVEL_POS[1],
        )
        self.mouse_controller.click(
            config.NEW_LEVEL_POS[0],
            config.NEW_LEVEL_POS[1],
            relative=True,
        )
        
        # Settle delay to allow transition popup to appear
        time.sleep(max(0.1, config.TRANSITION_RETRY_DELAY))

        logger.info(
            "Priority override: clicking level transition position at (%s, %s)",
            config.LEVEL_TRANSITION_POS[0],
            config.LEVEL_TRANSITION_POS[1],
        )
        self.mouse_controller.click(
            config.LEVEL_TRANSITION_POS[0],
            config.LEVEL_TRANSITION_POS[1],
            relative=True,
        )
        return True

    def _capture(self, max_y=None, force=False):
        cache_key = max_y if max_y is not None else "full"
        cached = self._capture_cache.get(cache_key)
        now = time.monotonic()
        if not force and cached and now - cached[0] <= self._capture_cache_ttl:
            return cached[1]

        with self._capture_lock:
            frame = self.window_capture.capture(max_y=max_y)
        
        if frame is None:
            logger.warning("Capture failed, returning empty frame placeholder")
            # Create a black frame of the expected size if capture fails
            frame = np.zeros((self.window_capture.target_height, self.window_capture.target_width, 3), dtype=np.uint8)
            if max_y is not None:
                frame = frame[:max_y, :]

        self._capture_cache[cache_key] = (now, frame)
        return frame

    def _clear_capture_cache(self):
        self._capture_cache.clear()
        self._new_level_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}
        self._new_level_red_icon_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}

    def _sleep_until(self, target_time):
        now = time.monotonic()
        if target_time <= now:
            return False

        interval = config.NEW_LEVEL_INTERRUPT_INTERVAL
        if interval <= 0:
            time.sleep(target_time - now)
            return False

        while now < target_time:
            remaining = target_time - now
            time.sleep(min(interval, remaining))
            if self._new_level_event.is_set():
                return True
            if self._should_interrupt_for_new_level(max_y=config.EXTENDED_SEARCH_Y, force=True, record_miss=False):
                return True
            now = time.monotonic()
        return False

    def _sleep_with_interrupt(self, duration):
        if duration <= 0:
            return False
        return self._sleep_until(time.monotonic() + duration)

    def _detect_new_level(self, screenshot=None, max_y=None, force=False, record_miss=True):
        target_max_y = max_y if max_y is not None else config.EXTENDED_SEARCH_Y
        now = time.monotonic()
        cached = self._new_level_cache
        if not force and cached["max_y"] == target_max_y and now - cached["timestamp"] <= self._capture_cache_ttl:
            return cached["result"]

        if screenshot is None:
            screenshot = self._capture(max_y=target_max_y, force=force)

        threshold = self.vision_optimizer.new_level_threshold if self.vision_optimizer.enabled else config.NEW_LEVEL_THRESHOLD
        result = self._find_new_level(screenshot, threshold=threshold)
        if result[0]:
            self.vision_optimizer.update_new_level_confidence(result[1])
        elif record_miss:
            self.vision_optimizer.update_new_level_miss()
        self._new_level_cache = {"timestamp": now, "result": result, "max_y": target_max_y}
        return result

    def _detect_new_level_red_icon(self, screenshot=None, max_y=None, force=False, record_miss=True):
        target_max_y = max_y if max_y is not None else config.EXTENDED_SEARCH_Y
        now = time.monotonic()
        cached = self._new_level_red_icon_cache
        cache_ttl = config.NEW_LEVEL_RED_ICON_CACHE_TTL
        if not force and cached["max_y"] == target_max_y and now - cached["timestamp"] <= cache_ttl:
            return cached["result"]

        if screenshot is None:
            screenshot = self._capture(max_y=target_max_y, force=force)

        height, width = screenshot.shape[:2]
        x_min = max(0, config.NEW_LEVEL_RED_ICON_X_MIN)
        x_max = min(width, config.NEW_LEVEL_RED_ICON_X_MAX)
        y_min = max(0, config.NEW_LEVEL_RED_ICON_Y_MIN)
        y_max = min(height, config.NEW_LEVEL_RED_ICON_Y_MAX)

        if x_min >= x_max or y_min >= y_max or not self.available_red_icon_templates:
            result = (False, 0.0, 0, 0)
            self._new_level_red_icon_cache = {
                "timestamp": now,
                "result": result,
                "max_y": target_max_y,
            }
            return result

        roi = screenshot[y_min:y_max, x_min:x_max]
        detections = {}
        buckets = {}
        template_hits = {}
        threshold = (
            self.vision_optimizer.new_level_red_icon_threshold
            if self.vision_optimizer.enabled
            else config.NEW_LEVEL_RED_ICON_THRESHOLD
        )

        for template_name, template, mask in self._iter_red_icon_templates():
            if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
                continue

            icons = self.image_matcher.find_all_templates(
                roi,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=80,
                template_name=template_name,
            )
            for conf, x, y in icons:
                abs_x = x + x_min
                abs_y = y + y_min
                if not self._passes_red_color_gate(screenshot, abs_x, abs_y):
                    continue
                self._merge_detection(
                    detections,
                    buckets,
                    abs_x,
                    abs_y,
                    template_name,
                    conf,
                )
                template_hits[template_name] = template_hits.get(template_name, 0) + 1

        min_matches = config.NEW_LEVEL_RED_ICON_MIN_MATCHES
        best_match = None
        for (x, y), matches in detections.items():
            if len(matches) >= min_matches:
                max_conf = max(conf for _, conf in matches)
                if best_match is None or max_conf > best_match[1]:
                    best_match = (True, max_conf, x, y)

        self._update_red_template_priority(template_hits)
        result = best_match or (False, 0.0, 0, 0)
        if result[0]:
            self.vision_optimizer.update_new_level_red_icon_confidence(result[1])
        elif record_miss:
            self.vision_optimizer.update_new_level_red_icon_miss()

        self._new_level_red_icon_cache = {"timestamp": now, "result": result, "max_y": target_max_y}
        return result

    def _detect_new_level_priority(self, screenshot=None, max_y=None, force=False, record_miss=True):
        red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
            record_miss=record_miss,
        )
        if red_found:
            self._mark_restaurant_completed("new level red icon", red_conf)
            return "new level red icon", red_conf, red_x, red_y

        found, confidence, x, y = self._detect_new_level(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
            record_miss=record_miss,
        )
        if found:
            self._mark_restaurant_completed("new level button", confidence)
            return "new level button", confidence, x, y

        return None

    def _should_interrupt_for_new_level(self, screenshot=None, max_y=None, force=False, record_miss=True):
        priority_hit = self._detect_new_level_priority(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
            record_miss=record_miss,
        )
        if priority_hit:
            source, confidence, x, y = priority_hit
            if source == "new level red icon":
                logger.info(
                    "Priority override: new level red icon detected at (%s, %s), interrupting current action",
                    x,
                    y,
                )
            else:
                logger.info("Priority override: new level detected, interrupting current action")
            return True
        return False

    def _mark_restaurant_completed(self, source, confidence=None):
        if self.completion_detected_time is not None:
            return
        self.completion_detected_time = datetime.now()
        self.completion_detected_by = source
        if confidence is None:
            logger.info("Restaurant completion detected via %s", source)
        else:
            logger.info("Restaurant completion detected via %s (confidence %.3f)", source, confidence)

    def _finalize_level_completion(self):
        self.total_levels_completed += 1

        time_spent = 0
        if self.current_level_start_time:
            completion_time = self.completion_detected_time or datetime.now()
            time_spent = (completion_time - self.current_level_start_time).total_seconds()

        completion_source = self.completion_detected_by or "new level trigger"
        self.current_level_start_time = datetime.now()
        self.completion_detected_time = None
        self.completion_detected_by = None

        self.telegram.notify_new_level(self.total_levels_completed, time_spent)
        self.historical_learner.record_completion(
            time_spent,
            completion_source,
        )

        logger.info(f"Level {self.total_levels_completed} completed. Time spent: {time_spent:.1f}s")
        logger.info("Transition finalized")

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

    def _has_stats_upgrade_icon(self, screenshot):
        if not self.red_icon_templates:
            return False, 0.0

        height, width = screenshot.shape[:2]
        x_min = max(0, config.UPGRADE_RED_ICON_X_MIN - config.STATS_ICON_PADDING)
        x_max = min(width, config.UPGRADE_RED_ICON_X_MAX + config.STATS_ICON_PADDING)
        y_min = max(0, config.UPGRADE_RED_ICON_Y_MIN - config.STATS_ICON_PADDING)
        y_max = min(height, config.UPGRADE_RED_ICON_Y_MAX + config.STATS_ICON_PADDING)

        if x_min >= x_max or y_min >= y_max:
            return False, 0.0

        roi = screenshot[y_min:y_max, x_min:x_max]
        threshold = (
            self.vision_optimizer.stats_upgrade_threshold
            if self.vision_optimizer.enabled
            else config.STATS_RED_ICON_THRESHOLD
        )
        best_confidence = 0.0
        template_hits = {}

        for template_name, template, mask in self._iter_red_icon_templates():
            icons = self.image_matcher.find_all_templates(
                roi,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=80,
                template_name=template_name,
            )

            if icons:
                for conf, x, y in icons:
                    abs_x = x + x_min
                    abs_y = y + y_min
                    if not self._passes_red_color_gate(screenshot, abs_x, abs_y):
                        continue
                    best_confidence = max(best_confidence, conf)
                    template_hits[template_name] = template_hits.get(template_name, 0) + 1

        self._update_red_template_priority(template_hits)
        return best_confidence > 0, best_confidence

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

    def _refine_template_position(
        self,
        template_name,
        expected_pos,
        search_radius,
        screenshot=None,
        threshold=None,
        check_color=False,
    ):
        if template_name not in self.templates:
            return expected_pos, False

        if screenshot is None:
            screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)

        template, mask = self.templates[template_name]
        x, y = expected_pos

        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(screenshot.shape[1], x + search_radius)
        y2 = min(screenshot.shape[0], y + search_radius)

        roi = screenshot[y1:y2, x1:x2]
        if roi.size == 0:
            return expected_pos, False

        found, confidence, rx, ry = self.image_matcher.find_template(
            roi,
            template,
            mask=mask,
            threshold=threshold,
            template_name=f"{template_name}-refine",
            check_color=check_color,
        )
        if not found:
            return expected_pos, False

        return (rx + x1, ry + y1), True

    def _refine_red_icon_position(self, x, y, screenshot=None):
        if not self.available_red_icon_templates:
            return (x, y), False, 0.0

        if screenshot is None:
            screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)

        search_radius = config.RED_ICON_REFINE_RADIUS
        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(screenshot.shape[1], x + search_radius)
        y2 = min(screenshot.shape[0], y + search_radius)

        roi = screenshot[y1:y2, x1:x2]
        if roi.size == 0:
            return (x, y), False, 0.0

        base_threshold = (
            self.vision_optimizer.red_icon_threshold
            if self.vision_optimizer.enabled
            else config.RED_ICON_THRESHOLD
        )
        threshold = max(0.0, base_threshold - config.RED_ICON_REFINE_THRESHOLD_DROP)
        best_match = None

        for template_name, template, mask in self._iter_red_icon_templates():
            found, confidence, rx, ry = self.image_matcher.find_template(
                roi,
                template,
                mask=mask,
                threshold=threshold,
                template_name=f"{template_name}-refine",
            )
            if not found:
                continue
            abs_x = rx + x1
            abs_y = ry + y1
            if not self._passes_red_color_gate(screenshot, abs_x, abs_y):
                continue
            if best_match is None or confidence > best_match[2]:
                best_match = (abs_x, abs_y, confidence)

        if best_match:
            return (best_match[0], best_match[1]), True, best_match[2]
        return (x, y), False, 0.0

    def _refine_upgrade_station_click_target(self, expected_pos, screenshot=None, threshold=None):
        refined_pos, refined = self._refine_template_position(
            "upgradeStation",
            expected_pos,
            config.UPGRADE_STATION_CLICK_REFINE_RADIUS,
            screenshot=screenshot,
            threshold=threshold,
            check_color=config.UPGRADE_STATION_COLOR_CHECK,
        )
        return refined_pos, refined

    def _detect_red_icons_in_view(self, screenshot, max_y=None, min_distance=80, threshold_override=None, min_matches_override=None):
        if not self.available_red_icon_templates:
            return []

        detections = {}
        buckets = {}
        template_hits = {}
        if max_y is not None:
            screenshot = screenshot[:max_y, :]
        base_threshold = (
            self.vision_optimizer.red_icon_threshold
            if self.vision_optimizer.enabled
            else config.RED_ICON_THRESHOLD
        )
        threshold = base_threshold if threshold_override is None else threshold_override

        for template_name, template, mask in self._iter_red_icon_templates():
            icons = self.image_matcher.find_all_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=min_distance,
                template_name=template_name,
            )

            for conf, x, y in icons:
                if not self._passes_red_color_gate(screenshot, x, y):
                    continue
                self._merge_detection(
                    detections,
                    buckets,
                    x,
                    y,
                    template_name,
                    conf,
                )
                template_hits[template_name] = template_hits.get(template_name, 0) + 1

        self._update_red_template_priority(template_hits)

        min_matches = config.RED_ICON_MIN_MATCHES if min_matches_override is None else min_matches_override
        red_icons = []
        for (x, y), matches in detections.items():
            if len(matches) >= min_matches:
                max_conf = max(conf for _, conf in matches)
                red_icons.append((max_conf, x, y))
        return red_icons

    def _is_red_icon_present_at(self, x, y, screenshot=None):
        if not self.available_red_icon_templates:
            return False

        target_screenshot = screenshot if screenshot is not None else self._capture(max_y=config.MAX_SEARCH_Y)

        if config.RED_ICON_COLOR_CHECK:
            if not self.image_matcher.is_red_dominant(
                target_screenshot,
                x,
                y,
                size=config.RED_ICON_COLOR_SAMPLE_SIZE,
                min_ratio=config.RED_ICON_COLOR_MIN_RATIO,
                min_mean=config.RED_ICON_COLOR_MIN_MEAN,
            ):
                return False

        threshold = (
            self.vision_optimizer.red_icon_threshold
            if self.vision_optimizer.enabled
            else config.RED_ICON_THRESHOLD
        )

        padding = config.RED_ICON_VERIFY_PADDING
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(target_screenshot.shape[1], x + padding)
        y2 = min(target_screenshot.shape[0], y + padding)

        roi = target_screenshot[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        for template_name, template, mask in self._iter_red_icon_templates():
            found, confidence, cx, cy = self.image_matcher.find_template(
                roi,
                template,
                mask=mask,
                threshold=threshold,
                template_name=f"{template_name}-verify",
            )
            if not found:
                continue

            abs_x = cx + x1
            abs_y = cy + y1
            if (
                abs(abs_x - x) <= config.RED_ICON_VERIFY_TOLERANCE
                and abs(abs_y - y) <= config.RED_ICON_VERIFY_TOLERANCE
            ):
                return True

        return False

    def _passes_red_color_gate(self, screenshot, x, y):
        if not config.RED_ICON_COLOR_CHECK:
            return True
        return self.image_matcher.is_red_dominant(
            screenshot,
            x,
            y,
            size=config.RED_ICON_COLOR_SAMPLE_SIZE,
            min_ratio=config.RED_ICON_COLOR_MIN_RATIO,
            min_mean=config.RED_ICON_COLOR_MIN_MEAN,
        )

    def _filter_forbidden_red_icons(self, red_icons):
        filtered_icons = []
        forbidden_icons = []
        forbidden_zones = self.forbidden_zones

        for conf, x, y in red_icons:
            in_forbidden = any(
                zone_x_min <= x <= zone_x_max and zone_y_min <= y <= zone_y_max
                for zone_x_min, zone_x_max, zone_y_min, zone_y_max in forbidden_zones
            )

            if in_forbidden:
                forbidden_icons.append((conf, x, y))
            else:
                filtered_icons.append((conf, x, y))

        return filtered_icons, forbidden_icons

    def _prioritize_red_icons(self, red_icons):
        def get_priority(icon):
            conf, x, y = icon
            for success_y in self.successful_red_icon_positions:
                if abs(y - success_y) < 50:
                    return (0, y)
            return (1, y)

        red_icons.sort(key=get_priority)

        max_per_scan = max(1, int(getattr(config, "RED_ICON_MAX_PER_SCAN", 1)))
        if len(red_icons) > max_per_scan:
            logger.debug(
                "Red icon queue limited from %s to %s for single-target interaction safety",
                len(red_icons),
                max_per_scan,
            )
            red_icons = red_icons[:max_per_scan]

        return red_icons

    def _build_scroll_segments(self, direction, distance_ratio=None):
        segments = max(1, int(getattr(config, "SCROLL_SEGMENTS", 1)))
        scroll_ratio = float(getattr(config, "SCROLL_DISTANCE_RATIO", 1.0)) if distance_ratio is None else float(distance_ratio)
        scroll_ratio = max(0.1, min(1.0, scroll_ratio))

        if direction == "up":
            full_start_x, full_start_y = config.SCROLL_UP_START_POS
            full_end_x, full_end_y = config.SCROLL_UP_END_POS
        else:
            full_start_x, full_start_y = config.SCROLL_DOWN_START_POS
            full_end_x, full_end_y = config.SCROLL_DOWN_END_POS

        end_x = int(full_start_x + (full_end_x - full_start_x) * scroll_ratio)
        end_y = int(full_start_y + (full_end_y - full_start_y) * scroll_ratio)

        min_segment_distance = max(1, int(getattr(config, "SCROLL_MIN_SEGMENT_DISTANCE", 1)))

        points = []
        pending_start = (full_start_x, full_start_y)

        for i in range(segments):
            t2 = (i + 1) / segments
            tx = int(full_start_x + (end_x - full_start_x) * t2)
            ty = int(full_start_y + (end_y - full_start_y) * t2)

            fx, fy = pending_start
            if abs(tx - fx) < min_segment_distance and abs(ty - fy) < min_segment_distance:
                continue

            points.append((fx, fy, tx, ty))
            pending_start = (tx, ty)

        if pending_start != (end_x, end_y):
            if points:
                last_fx, last_fy, _, _ = points[-1]
                points[-1] = (last_fx, last_fy, end_x, end_y)
            else:
                points.append((full_start_x, full_start_y, end_x, end_y))

        if not points:
            points.append((full_start_x, full_start_y, end_x, end_y))

        return points

    def _scroll_with_background_asset_detection(self, direction, scroll_duration, scan_for_red_icons=False, distance_ratio=None):
        stop_event = threading.Event()
        state_holder = {"state": None}

        def _interrupt_to_main_cycle(new_state, reason):
            logger.info("%s during scroll; interrupting scroll", reason)
            state_holder["state"] = new_state
            stop_event.set()

        def monitor_assets():
            interval = max(0.002, float(getattr(config, "SCROLL_ASSET_SCAN_INTERVAL", 0.008)))
            while not stop_event.is_set():
                screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)

                if self._should_interrupt_for_new_level(
                    screenshot=screenshot,
                    max_y=config.EXTENDED_SEARCH_Y,
                    record_miss=False,
                ):
                    _interrupt_to_main_cycle(State.TRANSITION_LEVEL, "New level detected")
                    return

                time.sleep(interval)

        monitor_thread = threading.Thread(target=monitor_assets, daemon=True)
        monitor_thread.start()

        segments = self._build_scroll_segments(direction, distance_ratio=distance_ratio)
        segment_duration = max(0.001, scroll_duration / max(1, len(segments)))
        segment_settle_delay = max(0.0, float(getattr(config, "SCROLL_SEGMENT_SETTLE_DELAY", 0.0)))

        for from_x, from_y, to_x, to_y in segments:
            if stop_event.is_set():
                break
            
            # Determine direction-specific duration and steps
            if direction == "up":
                final_duration = getattr(config, "SCROLL_UP_DURATION", segment_duration)
                final_steps = getattr(config, "SCROLL_UP_STEP_COUNT", None)
            else:
                final_duration = getattr(config, "SCROLL_DOWN_DURATION", segment_duration)
                final_steps = getattr(config, "SCROLL_DOWN_STEP_COUNT", None)
            
            # Re-calculate segment duration if using direction-specific total duration
            if direction == "up" and hasattr(config, "SCROLL_UP_DURATION"):
                final_seg_duration = max(0.001, config.SCROLL_UP_DURATION / max(1, len(segments)))
            elif direction == "down" and hasattr(config, "SCROLL_DOWN_DURATION"):
                final_seg_duration = max(0.001, config.SCROLL_DOWN_DURATION / max(1, len(segments)))
            else:
                final_seg_duration = segment_duration

            success = self.mouse_controller.drag(
                from_x,
                from_y,
                to_x,
                to_y,
                duration=final_seg_duration,
                relative=True,
                steps=final_steps,
                interrupt_callback=stop_event.is_set
            )
            if not success or stop_event.is_set():
                break
            if segment_settle_delay > 0:
                time.sleep(segment_settle_delay)
            
            # Synchronous Red Icon Scan (Step-Scan)
            if scan_for_red_icons:
                screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)
                scroll_threshold = max(
                    0.0,
                    (self.vision_optimizer.red_icon_threshold if self.vision_optimizer.enabled else config.RED_ICON_THRESHOLD)
                    - float(getattr(config, "SCROLL_RED_ICON_THRESHOLD_DROP", 0.04)),
                )
                scroll_icons = self._detect_red_icons_in_view(
                    screenshot,
                    max_y=config.MAX_SEARCH_Y,
                    min_distance=max(8, int(getattr(config, "SCROLL_RED_ICON_MIN_DISTANCE", 20))),
                    threshold_override=scroll_threshold,
                    min_matches_override=max(1, int(getattr(config, "SCROLL_RED_ICON_MIN_MATCHES", 1))),
                )
                
                if scroll_icons:
                    filtered_icons, forbidden_icons = self._filter_forbidden_red_icons(scroll_icons)
                    if forbidden_icons:
                        logger.info(f"Forbidden Zone Filter: {len(forbidden_icons)} icons removed during scroll")
                    if filtered_icons:
                        self.vision_optimizer.update_red_icon_scan([conf for conf, _, _ in scroll_icons])
                        self.red_icons = self._prioritize_red_icons(filtered_icons)
                        self.current_red_icon_index = 0
                        self.red_icon_cycle_count = 0
                        self.no_red_icons_found = False
                        self.work_done = True
                        logger.info(f"✓ {len(self.red_icons)} red icons detected during active scroll segment")
                        _interrupt_to_main_cycle(State.CLICK_RED_ICON, "Red icon detected (Step-Scan)")
                        break

        stop_event.set()
        monitor_thread.join(timeout=0.2)
        return state_holder["state"]

    def _scroll_and_scan_for_red_icons(self, direction, scroll_duration):
        if direction == "up":
            logger.info("⬆ Scroll UP (scan, segmented)")
        else:
            logger.info("⬇ Scroll DOWN (scan, segmented)")

        interrupt_state = self._scroll_with_background_asset_detection(
            direction,
            scroll_duration,
            scan_for_red_icons=True,
        )
        if interrupt_state is not None:
            return interrupt_state

        self._click_idle()

        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y, force=True)

        if self._should_interrupt_for_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
        ):
            return State.TRANSITION_LEVEL

        red_icons = self._detect_red_icons_in_view(screenshot, max_y=config.MAX_SEARCH_Y)
        red_icons = self._stable_red_icons(red_icons)
        self.vision_optimizer.update_red_icon_scan([conf for conf, _, _ in red_icons])
        if not red_icons:
            return None

        filtered_icons, forbidden_icons = self._filter_forbidden_red_icons(red_icons)
        if forbidden_icons:
            logger.info(f"Forbidden Zone Filter: {len(forbidden_icons)} icons removed during scroll")

        if not filtered_icons:
            return None

        self.red_icons = self._prioritize_red_icons(filtered_icons)
        self.current_red_icon_index = 0
        self.red_icon_cycle_count = 0
        self.no_red_icons_found = False
        self.work_done = True
        logger.info(f"✓ {len(self.red_icons)} red icons found during scroll; stopping scan")
        return State.CLICK_RED_ICON

    
    def load_templates(self):
        required_templates = self._required_template_names()
        scanner = AssetScanner(self.image_matcher)
        return scanner.scan(config.ASSETS_DIR, required_templates=required_templates)

    def _scan_and_click_non_red_assets(self, screenshot):
        clicked_targets = 0
        clicked_upgrade_station = False
        clicked_box = False

        upgrade_template = self.templates.get("upgradeStation")
        if upgrade_template is not None:
            template, mask = upgrade_template
            upgrade_threshold = (
                self.vision_optimizer.upgrade_station_threshold
                if self.vision_optimizer.enabled
                else config.UPGRADE_STATION_THRESHOLD
            )
            found, confidence, x, y = self.image_matcher.find_template(
                screenshot,
                template,
                mask=mask,
                threshold=upgrade_threshold,
                template_name="upgradeStation-no-red-fallback",
                check_color=config.UPGRADE_STATION_COLOR_CHECK,
            )
            if found:
                if self.mouse_controller.is_in_forbidden_zone(x, y):
                    logger.debug("Fallback scan: upgrade station in forbidden zone, skipping")
                else:
                    logger.info(
                        "Fallback scan: clicking upgrade station at (%s, %s) [%.2f%%]",
                        x,
                        y,
                        confidence * 100,
                    )
                    if self.mouse_controller.click(x, y, relative=True):
                        clicked_targets += 1
                        clicked_upgrade_station = True
                        self.upgrade_found_in_cycle = True
                        self.vision_optimizer.update_upgrade_station_confidence(confidence)

        for box_name in ("box1", "box2", "box3", "box4", "box5"):
            box_template = self.templates.get(box_name)
            if box_template is None:
                continue

            template, mask = box_template
            box_threshold = (
                self.vision_optimizer.box_threshold
                if self.vision_optimizer.enabled
                else config.BOX_THRESHOLD
            )
            
            # Find all matches for this box template
            matches = self.image_matcher.find_all_templates(
                screenshot,
                template,
                mask=mask,
                threshold=box_threshold,
                template_name=f"{box_name}-no-red-fallback",
                min_distance=30
            )

            min_matches = getattr(config, "BOX_MIN_MATCHES", 2)
            
            if len(matches) < min_matches:
                if not matches:
                    self.vision_optimizer.update_box_miss()
                continue

            # Use the best match
            confidence, x, y = max(matches, key=lambda m: m[0])

            if self.mouse_controller.is_in_forbidden_zone(x, y):
                logger.debug("Fallback scan: %s in forbidden zone, skipping", box_name)
                continue

            logger.info(
                "Fallback scan: clicking %s at (%s, %s) [%.2f%%]",
                box_name,
                x,
                y,
                confidence * 100,
            )
            if self.mouse_controller.click(x, y, relative=True):
                clicked_targets += 1
                clicked_box = True
                self.vision_optimizer.update_box_confidence(confidence)

        if clicked_targets > 0:
            self._no_red_scroll_cycle_pending = True
            logger.info(
                "Fallback scan summary: clicked %s target(s) [upgrade_station=%s, boxes=%s]; scheduling no-red scroll cycle",
                clicked_targets,
                clicked_upgrade_station,
                clicked_box,
            )

        return clicked_targets


    def _iter_red_icon_templates(self):
        if not self.available_red_icon_templates:
            return []

        if not self._red_template_priority:
            return self.available_red_icon_templates

        by_name = {name: (name, template, mask) for name, template, mask in self.available_red_icon_templates}
        ordered = []
        seen = set()

        for template_name in self._red_template_priority:
            item = by_name.get(template_name)
            if item is None:
                continue
            ordered.append(item)
            seen.add(template_name)

        for item in self.available_red_icon_templates:
            if item[0] in seen:
                continue
            ordered.append(item)

        return ordered

    def _update_red_template_priority(self, hit_counts):
        if not hit_counts:
            return

        now = time.monotonic()
        for template_name, count in hit_counts.items():
            self._red_template_hit_counts[template_name] = self._red_template_hit_counts.get(template_name, 0) + count
            self._red_template_last_seen[template_name] = now

        decay_window = max(1.0, float(getattr(config, "RED_ICON_STABILITY_CACHE_TTL", self._red_template_decay_window)))
        scored = []
        for name, count in self._red_template_hit_counts.items():
            last_seen = self._red_template_last_seen.get(name, now)
            age = max(0.0, now - last_seen)
            freshness = max(0.1, 1.0 - min(1.0, age / decay_window))
            score = count * freshness
            scored.append((name, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        limit = max(1, getattr(config, "RED_ICON_PRIORITY_TEMPLATE_LIMIT", 8))
        self._red_template_priority = [name for name, _ in scored[:limit]]

    def _build_available_red_icon_templates(self):
        available = []
        for template_name in self.red_icon_templates:
            if template_name in self.templates:
                template, mask = self.templates[template_name]
                available.append((template_name, template, mask))
        return available

    def get_runtime_behavior_snapshot(self):
        return {
            "click_delay": float(self.tuner.click_delay),
            "move_delay": float(self.tuner.move_delay),
            "upgrade_click_interval": float(self.tuner.upgrade_click_interval),
            "search_interval": float(self.tuner.search_interval),
        }

    def apply_learned_behavior(self, learned, reason="historical", best_time=0.0):
        if not learned:
            return
        self.tuner.click_delay = float(learned.get("click_delay", self.tuner.click_delay))
        self.tuner.move_delay = float(learned.get("move_delay", self.tuner.move_delay))
        self.tuner.upgrade_click_interval = float(
            learned.get("upgrade_click_interval", self.tuner.upgrade_click_interval)
        )
        self.tuner.search_interval = float(learned.get("search_interval", self.tuner.search_interval))
        logger.info(
            "Historical learner (%s) applied timing profile from best %.2fs run",
            reason,
            best_time,
        )
        self._apply_tuning()

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
        self._click_idle()

        self.work_done = False
        self.forbidden_icon_scrolls = 0
        
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]

        if self._should_interrupt_for_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
            force=True,
        ):
            logger.info("New level detected during scan, transitioning")
            return State.TRANSITION_LEVEL

        self.red_icons = self._detect_red_icons_in_view(
            screenshot,
            max_y=config.MAX_SEARCH_Y,
        )
        self.red_icons = self._stable_red_icons(self.red_icons)

        self.vision_optimizer.update_red_icon_scan([conf for conf, _, _ in self.red_icons])

        logger.info(
            "Red Icon Detection: %s valid icons found (min %s template matches)",
            len(self.red_icons),
            config.RED_ICON_MIN_MATCHES,
        )

        red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
            force=True,
        )
        if red_found:
            self._mark_restaurant_completed("new level red icon", red_conf)
            logger.info(f"New level detected! Red icon at ({red_x}, {red_y})")
            return State.CHECK_NEW_LEVEL
        
        if not self.red_icons:
            clicked_targets = self._scan_and_click_non_red_assets(limited_screenshot)
            if clicked_targets > 0:
                logger.info(
                    "No red icons detected; fallback clicked %s non-red assets. Proceeding to no-icon scrolling cycle.",
                    clicked_targets,
                )
                self.no_red_icons_found = True
                return State.SCROLL

            logger.info("No valid red icons after scan; no fallback assets found, scrolling to search")
            self.no_red_icons_found = True
            return State.SCROLL
        else:
            filtered_icons, forbidden_icons = self._filter_forbidden_red_icons(self.red_icons)
            if forbidden_icons:
                logger.info(f"Forbidden Zone Filter: {len(forbidden_icons)} icons removed")
            
            if not filtered_icons:
                if forbidden_icons:
                    logger.info(f"0 Safe / {len(forbidden_icons)} Forbidden icons detected → using main scroll search")
                else:
                    logger.info("0 Safe icons detected → using main scroll search")
                
                clicked_targets = self._scan_and_click_non_red_assets(limited_screenshot)
                if clicked_targets > 0:
                    logger.info(f"Fallback success: clicked {clicked_targets} non-red assets. Proceeding to scroll.")
                    self.no_red_icons_found = True
                    return State.SCROLL

                logger.info("No clickable red icons or fallback assets found; scrolling to search area")
                self.no_red_icons_found = True
                return State.SCROLL
            
            # MATRIX CASE: 1+ Safe icons detected -> Proceed to Click
            self.red_icons = self._prioritize_red_icons(filtered_icons)
            logger.info(f"✓ {len(self.red_icons)} Safe red icons ready to process")
            self.current_red_icon_index = 0
            self.red_icon_cycle_count = 0
            self.work_done = True
            self.no_red_icons_found = False
            return State.CLICK_RED_ICON
    
    def handle_click_red_icon(self, current_state):
        if self.current_red_icon_index >= len(self.red_icons):
            logger.info("All red icons processed, continuing cycle")
            return State.OPEN_BOXES
        
        confidence, x, y = self.red_icons[self.current_red_icon_index]
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y, force=True)
        limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]

        if self._should_interrupt_for_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
            force=True,
        ):
            return State.TRANSITION_LEVEL

        if not self._is_red_icon_present_at(x, y, screenshot=limited_screenshot):
            logger.info(
                "Red icon no longer present at (%s, %s); skipping click",
                x,
                y,
            )
            self.current_red_icon_index += 1
            if self.current_red_icon_index < len(self.red_icons):
                return State.CLICK_RED_ICON
            return State.FIND_RED_ICONS

        refined_pos, refined, refined_conf = self._refine_red_icon_position(
            x,
            y,
            screenshot=limited_screenshot,
        )
        if refined:
            x, y = refined_pos
            self.vision_optimizer.update_red_icon_confidences([refined_conf])

        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y
        
        if self.mouse_controller.is_in_forbidden_zone(click_x, click_y):
            logger.warning(f"Red icon click blocked - position with offset ({click_x}, {click_y}) is in forbidden zone")
            self.current_red_icon_index += 1
            return State.CLICK_RED_ICON if self.current_red_icon_index < len(self.red_icons) else State.OPEN_BOXES
        
        logger.info(f"Clicking red icon {self.current_red_icon_index + 1}/{len(self.red_icons)} at ({click_x}, {click_y})")
        click_success = self.mouse_controller.click(click_x, click_y, relative=True)
        self.tuner.record_click_result(click_success)
        self._apply_tuning()
        
        self.red_icon_cycle_count = 0
        return State.CHECK_UNLOCK
    
    def handle_check_unlock(self, current_state):
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]
        
        if self._should_interrupt_for_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
            force=True,
        ):
            return State.TRANSITION_LEVEL

        clicked_unlock = False
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
                    clicked_unlock = self.mouse_controller.click(x, y, relative=True)

        if clicked_unlock:
            if self._sleep_with_interrupt(config.STATE_DELAY):
                return State.TRANSITION_LEVEL
            return State.SEARCH_UPGRADE_STATION

        return State.SEARCH_UPGRADE_STATION
    
    def handle_search_upgrade_station(self, current_state):
        max_attempts = 5
        base_threshold = (
            self.vision_optimizer.upgrade_station_threshold
            if self.vision_optimizer.enabled
            else config.UPGRADE_STATION_THRESHOLD
        )
        relaxed_threshold = base_threshold - 0.05
        retry_delay = self.tuner.search_interval
        
        for attempt in range(max_attempts):
            screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
            limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]
            
            if self._should_interrupt_for_new_level(
                screenshot=screenshot,
                max_y=config.EXTENDED_SEARCH_Y,
                force=True,
            ):
                return State.TRANSITION_LEVEL

            if "upgradeStation" in self.templates:
                template, mask = self.templates["upgradeStation"]
                
                current_threshold = base_threshold if attempt < 2 else relaxed_threshold
                
                found, confidence, x, y = self.image_matcher.find_template(
                    limited_screenshot, template, mask=mask,
                    threshold=current_threshold, template_name="upgradeStation"
                )
                
                if found:
                    logger.info(f"✓ Upgrade station found (attempt {attempt + 1})")
                    refined_pos, refined = self._refine_template_position(
                        "upgradeStation",
                        (x, y),
                        config.UPGRADE_STATION_REFINE_RADIUS,
                        screenshot=limited_screenshot,
                        threshold=current_threshold,
                        check_color=config.UPGRADE_STATION_COLOR_CHECK,
                    )
                    self.upgrade_station_pos = refined_pos
                    if refined:
                        logger.debug(
                            "Refined upgrade station position: (%s, %s) -> (%s, %s)",
                            x,
                            y,
                            refined_pos[0],
                            refined_pos[1],
                        )
                    self.vision_optimizer.update_upgrade_station_confidence(confidence)
                    
                    if self.current_red_icon_index < len(self.red_icons):
                        _, _, red_y = self.red_icons[self.current_red_icon_index]
                        if red_y not in self.successful_red_icon_positions:
                            self.successful_red_icon_positions.append(red_y)
                    
                    self.upgrade_found_in_cycle = True
                    self.consecutive_failed_cycles = 0
                    self._last_upgrade_station_pos = self.upgrade_station_pos
                    self.tuner.record_search_result(True)
                    self._apply_tuning()
                    return State.HOLD_UPGRADE_STATION
            
            if attempt < max_attempts - 1:
                if retry_delay > 0 and self._sleep_with_interrupt(retry_delay):
                    return State.TRANSITION_LEVEL
        
        logger.info(f"✗ Upgrade station not found (failed cycles: {self.consecutive_failed_cycles + 1})")
        self.vision_optimizer.update_upgrade_station_miss()
        self.tuner.record_search_result(False)
        self._apply_tuning()
        self.red_icon_processed_count += 1
        self.consecutive_failed_cycles += 1
        self.current_red_icon_index += 1
        if self.current_red_icon_index < len(self.red_icons):
            logger.info("Trying next red icon after station search miss")
            return State.CLICK_RED_ICON
        return State.OPEN_BOXES
    
    def handle_hold_upgrade_station(self, current_state):
        base_pos = self.upgrade_station_pos
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y, force=True)
        limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]

        if self._should_interrupt_for_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
            force=True,
        ):
            return State.TRANSITION_LEVEL

        hold_threshold = (
            self.vision_optimizer.upgrade_station_threshold
            if self.vision_optimizer.enabled
            else config.UPGRADE_STATION_THRESHOLD
        )
        refined_pos, refined = self._refine_template_position(
            "upgradeStation",
            base_pos,
            config.UPGRADE_STATION_REFINE_RADIUS,
            screenshot=limited_screenshot,
            threshold=hold_threshold,
            check_color=config.UPGRADE_STATION_COLOR_CHECK,
        )
        x, y = refined_pos
        if refined:
            self._last_upgrade_station_pos = refined_pos
            self.upgrade_station_pos = refined_pos
        elif self._last_upgrade_station_pos:
            last_x, last_y = self._last_upgrade_station_pos
            drift_limit = config.UPGRADE_STATION_REFINE_RADIUS * 2
            if abs(last_x - base_pos[0]) <= drift_limit and abs(last_y - base_pos[1]) <= drift_limit:
                x, y = self._last_upgrade_station_pos
                self.upgrade_station_pos = self._last_upgrade_station_pos

        click_refined_pos, click_refined = self._refine_upgrade_station_click_target(
            (x, y),
            screenshot=limited_screenshot,
            threshold=hold_threshold,
        )
        if click_refined:
            x, y = click_refined_pos
            self._last_upgrade_station_pos = click_refined_pos
            self.upgrade_station_pos = click_refined_pos

        if self.mouse_controller.is_in_forbidden_zone(x, y):
            logger.warning("Upgrade station position is in forbidden zone; skipping clicks")
            self.red_icon_processed_count += 1
            self.current_red_icon_index += 1
            if self.current_red_icon_index < len(self.red_icons):
                return State.CLICK_RED_ICON
            return State.OPEN_BOXES
        
        logger.info("Holding upgrade station click...")

        max_hold_time = config.UPGRADE_HOLD_DURATION
        check_interval = max(config.UPGRADE_CHECK_INTERVAL, self.tuner.search_interval)
        start_time = time.monotonic()
        end_time = start_time + max_hold_time
        upgrade_missing_logged = False
        next_check_time = start_time + check_interval
        interrupt_state = None

        if not self.mouse_controller.mouse_down(x, y, relative=True):
            self.red_icon_processed_count += 1
            self.current_red_icon_index += 1
            if self.current_red_icon_index < len(self.red_icons):
                return State.CLICK_RED_ICON
            return State.OPEN_BOXES

        try:
            while True:
                now = time.monotonic()
                if now >= end_time:
                    break

                if now >= next_check_time:
                    screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y, force=True)
                    limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]

                    if "upgradeStation" in self.templates:
                        template, mask = self.templates["upgradeStation"]
                        found, confidence, found_x, found_y = self.image_matcher.find_template(
                            limited_screenshot, template, mask=mask,
                            threshold=hold_threshold, template_name="upgradeStation",
                            check_color=config.UPGRADE_STATION_COLOR_CHECK
                        )

                        if not found and not upgrade_missing_logged:
                            logger.info("Upgrade station not found while holding; continuing until duration completes.")
                            upgrade_missing_logged = True

                    if self._should_interrupt_for_new_level(
                        screenshot=screenshot,
                        max_y=config.EXTENDED_SEARCH_Y,
                        force=True,
                    ):
                        interrupt_state = State.TRANSITION_LEVEL
                        break

                    next_check_time = max(next_check_time + check_interval, now + check_interval)

                now = time.monotonic()
                next_action_time = min(next_check_time, end_time)
                if self._sleep_until(next_action_time):
                    interrupt_state = State.TRANSITION_LEVEL
                    break
        finally:
            self.mouse_controller.mouse_up(x, y, relative=True)

        elapsed_time = time.monotonic() - start_time
        logger.info(f"Clicking complete: hold duration {elapsed_time:.1f}s")
        
        self._click_idle()
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL
        
        self.red_icon_processed_count += 1
        self.current_red_icon_index += 1

        logger.info("✓ Upgrade station complete → Stats upgrade next")
        return interrupt_state or State.UPGRADE_STATS
    
    def handle_upgrade_stats(self, current_state):
        logger.info("⬆ Stats upgrade starting")
        self._click_idle()
        
        extended_screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = extended_screenshot[:config.MAX_SEARCH_Y, :]

        found, confidence, x, y = self._detect_new_level(
            screenshot=extended_screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
        )
        if found:
            logger.info("New level detected during stats upgrade")
            return State.TRANSITION_LEVEL
        
        has_stats_icon, stats_confidence = self._has_stats_upgrade_icon(extended_screenshot)
        if not has_stats_icon:
            logger.info("✗ No stats icon, skipping")
            self.vision_optimizer.update_stats_upgrade_miss()
            return State.SCROLL

        self.vision_optimizer.update_stats_upgrade_confidence(stats_confidence)
        
        logger.info("✓ Stats icon found, upgrading")
        self.mouse_controller.click(config.STATS_UPGRADE_BUTTON_POS[0], config.STATS_UPGRADE_BUTTON_POS[1], relative=True)
        if self._sleep_with_interrupt(config.STATE_DELAY):
            return State.TRANSITION_LEVEL
        
        start_time = time.monotonic()
        last_new_level_check = 0.0
        next_click_time = start_time
        while time.monotonic() - start_time < config.STATS_UPGRADE_CLICK_DURATION:
            if self._sleep_until(next_click_time):
                return State.TRANSITION_LEVEL
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
                screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
                if self._should_interrupt_for_new_level(
                    screenshot=screenshot,
                    max_y=config.EXTENDED_SEARCH_Y,
                ):
                    return State.TRANSITION_LEVEL
                last_new_level_check = elapsed
        
        self._click_idle()
        logger.info("========== STAT UPGRADE COMPLETED ==========")
        return State.OPEN_BOXES
    
    def handle_open_boxes(self, current_state):
        self._click_idle()
        
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[:config.MAX_SEARCH_Y, :]

        found, confidence, x, y = self._detect_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
        )
        if found:
            logger.info("New level found, transitioning")
            return State.TRANSITION_LEVEL
        
        box_names = ["box1", "box2", "box3", "box4", "box5"]
        boxes_found = 0
        
        for box_name in box_names:
            if box_name in self.templates:
                template, mask = self.templates[box_name]
                box_threshold = (
                    self.vision_optimizer.box_threshold
                    if self.vision_optimizer.enabled
                    else config.BOX_THRESHOLD
                )
                
                # Find all potential matches for this box template
                matches = self.image_matcher.find_all_templates(
                    limited_screenshot, 
                    template, 
                    mask=mask,
                    threshold=box_threshold, 
                    template_name=box_name,
                    min_distance=30
                )
                
                # Minimum match check (like red icons)
                min_matches = getattr(config, "BOX_MIN_MATCHES", 2)
                
                if len(matches) >= min_matches:
                    # Use the best match among the results
                    confidence, x, y = max(matches, key=lambda m: m[0])
                    
                    if self.mouse_controller.is_in_forbidden_zone(x, y):
                        logger.debug(f"{box_name} in forbidden zone, skipping")
                    else:
                        self.mouse_controller.click(x, y, relative=True)
                        boxes_found += 1
                        self.vision_optimizer.update_box_confidence(confidence)
                else:
                    if not matches:
                        self.vision_optimizer.update_box_miss()
        
        if self._should_interrupt_for_new_level(
            max_y=config.EXTENDED_SEARCH_Y,
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
        self._click_idle()
        
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)

        found, confidence, x, y = self._detect_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
        )
        if found:
            logger.info("New level detected before scroll")
            return State.TRANSITION_LEVEL
        
        scroll_duration = config.NO_ICON_SCROLL_DURATION if self.no_red_icons_found else config.SCROLL_DURATION

        direction = 'up' if self.scroll_direction == 'up' else 'down'
        
        if self.no_red_icons_found:
            max_count = config.NO_ICON_SCROLL_UP_COUNT if direction == 'up' else config.NO_ICON_SCROLL_DOWN_COUNT
            logger.info(f"No red icons found → searching {direction.upper()} ({self.scroll_count + 1}/{max_count})")
        else:
            max_count = getattr(config, "NORMAL_SCROLL_UP_COUNT" if direction == 'up' else "NORMAL_SCROLL_DOWN_COUNT", self.max_scroll_count)
            logger.info(f"{'⬆' if direction == 'up' else '⬇'} Scroll {direction.upper()} ({self.scroll_count + 1}/{max_count}, segmented)")

        interrupt_state = self._scroll_with_background_asset_detection(
            direction,
            scroll_duration,
            scan_for_red_icons=True,
        )
        
        # Always increment scroll count if we performed a drag action, 
        # even if interrupted, to ensure we eventually swap directions.
        self.scroll_count += 1
        
        if self.scroll_count >= max_count:
            logger.info(f"Reached max scrolls ({max_count}) for direction {direction.upper()} → swapping")
            self.scroll_direction = 'down' if self.scroll_direction == 'up' else 'up'
            self.scroll_count = 0

        if interrupt_state is not None:
            return interrupt_state

        self._click_idle()
        return State.FIND_RED_ICONS
    
    def handle_check_new_level(self, current_state):
        self._click_idle()
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL

        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        if self._should_interrupt_for_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
        ):
            return State.TRANSITION_LEVEL
        
        logger.info("Clicking new level position")
        self.mouse_controller.click(config.NEW_LEVEL_POS[0], config.NEW_LEVEL_POS[1], relative=True)
        if config.NEW_LEVEL_BUTTON_DELAY > 0:
            if self._sleep_with_interrupt(config.NEW_LEVEL_BUTTON_DELAY):
                return State.TRANSITION_LEVEL
        
        logger.info("Triggering transition click (Fly to New City)")
        self.mouse_controller.click(config.LEVEL_TRANSITION_POS[0], config.LEVEL_TRANSITION_POS[1], relative=True)
        if config.NEW_LEVEL_FOLLOWUP_DELAY > 0:
            if self._sleep_with_interrupt(config.NEW_LEVEL_FOLLOWUP_DELAY):
                return State.TRANSITION_LEVEL

        self._finalize_level_completion()
        return State.WAIT_FOR_UNLOCK
    
    def handle_transition_level(self, current_state):
        self._click_idle()
        
        max_attempts = 5
        
        for attempt in range(max_attempts):
            screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)

            found, confidence, x, y = self._detect_new_level(
                screenshot=screenshot,
                max_y=config.EXTENDED_SEARCH_Y,
            )
            if found:
                self._mark_restaurant_completed("new level button", confidence)
                logger.info(f"New level button found (attempt {attempt + 1})")
                
                # Clicking NEW_LEVEL_POS as requested by the user
                logger.info("Clicking new level position")
                self.mouse_controller.click(config.NEW_LEVEL_POS[0], config.NEW_LEVEL_POS[1], relative=True)
                if config.TRANSITION_POST_CLICK_DELAY > 0:
                    if self._sleep_with_interrupt(config.TRANSITION_POST_CLICK_DELAY):
                        return State.TRANSITION_LEVEL

                # Click the transition button (Fly to New City) which appears after clicking the new level button
                logger.info("Clicking level transition button (Fly to New City)")
                self.mouse_controller.click(config.LEVEL_TRANSITION_POS[0], config.LEVEL_TRANSITION_POS[1], relative=True)
                if config.TRANSITION_POST_CLICK_DELAY > 0:
                    if self._sleep_with_interrupt(config.TRANSITION_POST_CLICK_DELAY):
                        return State.TRANSITION_LEVEL

                self._finalize_level_completion()
                logger.info("Waiting for unlock button after level transition")
                return State.WAIT_FOR_UNLOCK
            
            if attempt < max_attempts - 1:
                if config.TRANSITION_RETRY_DELAY > 0:
                    if self._sleep_with_interrupt(config.TRANSITION_RETRY_DELAY):
                        return State.TRANSITION_LEVEL
        
        logger.warning("New level button not found after 5 attempts")
        self.scroll_direction = 'up'
        self.scroll_count = 0
        return State.FIND_RED_ICONS
    
    def handle_wait_for_unlock(self, current_state):
        self._click_idle()
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL
        
        self.wait_for_unlock_attempts += 1
        logger.debug(f"Waiting for unlock button (attempt {self.wait_for_unlock_attempts}/{self.max_wait_for_unlock_attempts})")
        
        if self.wait_for_unlock_attempts > self.max_wait_for_unlock_attempts:
            logger.warning(f"Unlock button not found after {self.max_wait_for_unlock_attempts} attempts, resetting to scroll")
            self.wait_for_unlock_attempts = 0
            self.scroll_direction = 'down'
            self.scroll_count = 0
            return State.FIND_RED_ICONS
        
        screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)

        if self._should_interrupt_for_new_level(
            screenshot=screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
            force=True,
        ):
            return State.TRANSITION_LEVEL

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
                    if self._sleep_with_interrupt(config.UNLOCK_POST_CLICK_DELAY):
                        return State.TRANSITION_LEVEL
                logger.info("Starting new level")
                self.wait_for_unlock_attempts = 0
                self.scroll_direction = 'down'
                self.scroll_count = 0
                return State.FIND_RED_ICONS
        
        if config.WAIT_UNLOCK_RETRY_DELAY > 0:
            if self._sleep_with_interrupt(config.WAIT_UNLOCK_RETRY_DELAY):
                return State.TRANSITION_LEVEL
        return State.WAIT_FOR_UNLOCK
    
    def run(self):
        self.running = True
        logger.info("Bot started - Press Ctrl+C to stop")
        if self.current_level_start_time is None:
            self.current_level_start_time = datetime.now()
            logger.info("Starting level timer at bot start")

        if self._new_level_monitor_thread is None or not self._new_level_monitor_thread.is_alive():
            self._new_level_monitor_stop.clear()
            self._new_level_monitor_thread = threading.Thread(
                target=self._monitor_new_level,
                name="new_level_monitor",
                daemon=True,
            )
            self._new_level_monitor_thread.start()

        self.historical_learner.start()
        
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
        self._enforce_state_min_interval()
        self.state_machine.update()
    
    def stop(self):
        self.running = False
        self._new_level_monitor_stop.set()
        if self._new_level_monitor_thread and self._new_level_monitor_thread.is_alive():
            self._new_level_monitor_thread.join(timeout=1.0)
        self.historical_learner.stop()
        if self.overlay:
            self.overlay.stop()
        logger.info("Bot stopped")
