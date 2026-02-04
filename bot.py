import json
import os
import time
import logging
import threading
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


class VisionOptimizer:
    def __init__(self, persistence=None):
        self.enabled = config.AI_VISION_ENABLED
        self.alpha = config.AI_VISION_ALPHA
        self.alpha_max = config.AI_VISION_ALPHA_MAX
        self.confidence_boost = config.AI_VISION_CONFIDENCE_BOOST
        self.red_icon_threshold = config.RED_ICON_THRESHOLD
        self.new_level_threshold = config.NEW_LEVEL_THRESHOLD
        self.new_level_red_icon_threshold = config.NEW_LEVEL_RED_ICON_THRESHOLD
        self.upgrade_station_threshold = config.UPGRADE_STATION_THRESHOLD
        self.stats_upgrade_threshold = config.STATS_RED_ICON_THRESHOLD
        self.persistence = persistence
        self._red_icon_miss_count = 0
        self._new_level_miss_count = 0
        self._new_level_red_icon_miss_count = 0
        self._upgrade_station_miss_count = 0
        self._stats_upgrade_miss_count = 0

    def _ema(self, current, new_value, alpha=None):
        blend = self.alpha if alpha is None else alpha
        return (1 - blend) * current + blend * new_value

    def _adaptive_alpha(self, confidence):
        if confidence <= 0:
            return self.alpha
        boost = max(0.0, min(1.0, (confidence - 0.8))) * self.confidence_boost
        return min(self.alpha + boost, self.alpha_max)

    def update_red_icon_confidences(self, confidences):
        if not self.enabled or not confidences:
            return
        avg_conf = sum(confidences) / len(confidences)
        target = max(
            config.AI_RED_ICON_THRESHOLD_MIN,
            min(avg_conf - config.AI_RED_ICON_MARGIN, config.AI_RED_ICON_THRESHOLD_MAX),
        )
        self.red_icon_threshold = self._ema(self.red_icon_threshold, target, self._adaptive_alpha(avg_conf))
        self._persist()

    def update_red_icon_scan(self, confidences):
        if not self.enabled:
            return
        if confidences:
            self._red_icon_miss_count = 0
            self.update_red_icon_confidences(confidences)
            return

        self._red_icon_miss_count += 1
        if self._red_icon_miss_count < config.AI_RED_ICON_MISS_WINDOW:
            return
        self._red_icon_miss_count = 0

        target = max(
            config.AI_RED_ICON_THRESHOLD_MIN,
            self.red_icon_threshold - config.AI_RED_ICON_MISS_STEP,
        )
        self.red_icon_threshold = self._ema(self.red_icon_threshold, target, self.alpha_max)
        self._persist()

    def update_new_level_confidence(self, confidence):
        if not self.enabled or confidence <= 0:
            return
        self._new_level_miss_count = 0
        target = max(
            config.AI_NEW_LEVEL_THRESHOLD_MIN,
            min(confidence, config.AI_NEW_LEVEL_THRESHOLD_MAX),
        )
        self.new_level_threshold = self._ema(
            self.new_level_threshold,
            target,
            self._adaptive_alpha(confidence),
        )
        self._persist()

    def update_new_level_red_icon_confidence(self, confidence):
        if not self.enabled or confidence <= 0:
            return
        self._new_level_red_icon_miss_count = 0
        target = max(
            config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN,
            min(confidence, config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX),
        )
        self.new_level_red_icon_threshold = self._ema(
            self.new_level_red_icon_threshold,
            target,
            self._adaptive_alpha(confidence),
        )
        self._persist()

    def update_upgrade_station_confidence(self, confidence):
        if not self.enabled or confidence <= 0:
            return
        self._upgrade_station_miss_count = 0
        target = max(
            config.AI_UPGRADE_STATION_THRESHOLD_MIN,
            min(confidence, config.AI_UPGRADE_STATION_THRESHOLD_MAX),
        )
        self.upgrade_station_threshold = self._ema(
            self.upgrade_station_threshold,
            target,
            self._adaptive_alpha(confidence),
        )
        self._persist()

    def update_stats_upgrade_confidence(self, confidence):
        if not self.enabled or confidence <= 0:
            return
        self._stats_upgrade_miss_count = 0
        target = max(
            config.AI_STATS_UPGRADE_THRESHOLD_MIN,
            min(confidence, config.AI_STATS_UPGRADE_THRESHOLD_MAX),
        )
        self.stats_upgrade_threshold = self._ema(
            self.stats_upgrade_threshold,
            target,
            self._adaptive_alpha(confidence),
        )
        self._persist()

    def update_new_level_miss(self):
        if not self.enabled:
            return
        self._new_level_miss_count += 1
        if self._new_level_miss_count < config.AI_NEW_LEVEL_MISS_WINDOW:
            return
        self._new_level_miss_count = 0
        target = max(
            config.AI_NEW_LEVEL_THRESHOLD_MIN,
            self.new_level_threshold - config.AI_NEW_LEVEL_MISS_STEP,
        )
        self.new_level_threshold = self._ema(self.new_level_threshold, target, self.alpha_max)
        self._persist()

    def update_new_level_red_icon_miss(self):
        if not self.enabled:
            return
        self._new_level_red_icon_miss_count += 1
        if self._new_level_red_icon_miss_count < config.AI_NEW_LEVEL_RED_ICON_MISS_WINDOW:
            return
        self._new_level_red_icon_miss_count = 0
        target = max(
            config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN,
            self.new_level_red_icon_threshold - config.AI_NEW_LEVEL_RED_ICON_MISS_STEP,
        )
        self.new_level_red_icon_threshold = self._ema(
            self.new_level_red_icon_threshold,
            target,
            self.alpha_max,
        )
        self._persist()

    def update_upgrade_station_miss(self):
        if not self.enabled:
            return
        self._upgrade_station_miss_count += 1
        if self._upgrade_station_miss_count < config.AI_UPGRADE_STATION_MISS_WINDOW:
            return
        self._upgrade_station_miss_count = 0
        target = max(
            config.AI_UPGRADE_STATION_THRESHOLD_MIN,
            self.upgrade_station_threshold - config.AI_UPGRADE_STATION_MISS_STEP,
        )
        self.upgrade_station_threshold = self._ema(
            self.upgrade_station_threshold,
            target,
            self.alpha_max,
        )
        self._persist()

    def update_stats_upgrade_miss(self):
        if not self.enabled:
            return
        self._stats_upgrade_miss_count += 1
        if self._stats_upgrade_miss_count < config.AI_STATS_UPGRADE_MISS_WINDOW:
            return
        self._stats_upgrade_miss_count = 0
        target = max(
            config.AI_STATS_UPGRADE_THRESHOLD_MIN,
            self.stats_upgrade_threshold - config.AI_STATS_UPGRADE_MISS_STEP,
        )
        self.stats_upgrade_threshold = self._ema(
            self.stats_upgrade_threshold,
            target,
            self.alpha_max,
        )
        self._persist()

    def apply_persisted_state(self, state):
        if not state:
            return
        if "red_icon_threshold" in state:
            self.red_icon_threshold = float(state["red_icon_threshold"])
        if "new_level_threshold" in state:
            self.new_level_threshold = float(state["new_level_threshold"])
        if "new_level_red_icon_threshold" in state:
            self.new_level_red_icon_threshold = float(state["new_level_red_icon_threshold"])
        if "upgrade_station_threshold" in state:
            self.upgrade_station_threshold = float(state["upgrade_station_threshold"])
        if "stats_upgrade_threshold" in state:
            self.stats_upgrade_threshold = float(state["stats_upgrade_threshold"])

    def _persist(self):
        if not self.persistence:
            return
        state = {
            "red_icon_threshold": self.red_icon_threshold,
            "new_level_threshold": self.new_level_threshold,
            "new_level_red_icon_threshold": self.new_level_red_icon_threshold,
            "upgrade_station_threshold": self.upgrade_station_threshold,
            "stats_upgrade_threshold": self.stats_upgrade_threshold,
        }
        self.persistence.save({key: float(value) for key, value in state.items()})


class VisionPersistence:
    def __init__(self, path, save_interval):
        self.path = path
        self.save_interval = save_interval
        self._last_save_time = 0.0

    def load(self):
        if not self.path:
            return {}
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to load vision state from %s: %s. Using defaults.",
                self.path,
                exc,
            )
            return {}

    def save(self, state):
        if not self.path:
            return
        now = time.monotonic()
        if self.save_interval > 0 and now - self._last_save_time < self.save_interval:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        self._last_save_time = now


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
        self._capture_cache = {}
        self._capture_cache_ttl = config.CAPTURE_CACHE_TTL
        self._new_level_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}
        self._new_level_red_icon_cache = {"timestamp": 0.0, "result": (False, 0.0, 0, 0), "max_y": None}
        self._capture_lock = threading.Lock()
        self._new_level_event = threading.Event()
        self._new_level_interrupt = None
        self._new_level_monitor_stop = threading.Event()
        self._new_level_monitor_thread = None
        self._last_upgrade_station_pos = None
        self._last_new_level_override_time = 0.0

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
        if not self._new_level_event.is_set():
            return None
        interrupt = self._new_level_interrupt
        self._new_level_event.clear()
        return interrupt

    def _monitor_new_level(self):
        interval = config.NEW_LEVEL_MONITOR_INTERVAL
        while not self._new_level_monitor_stop.is_set():
            if self._new_level_event.is_set():
                time.sleep(max(interval, 0.01))
                continue

            limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)

            red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
                screenshot=limited_screenshot,
                max_y=config.MAX_SEARCH_Y,
                force=True,
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
                max_y=config.MAX_SEARCH_Y,
                force=True,
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
        interrupt = self._consume_new_level_interrupt()
        if interrupt:
            logger.info(
                "Priority override: background %s detected at (%s, %s), interrupting current action",
                interrupt["source"],
                interrupt["x"],
                interrupt["y"],
            )
            self._click_new_level_override(source=interrupt["source"])
            return State.TRANSITION_LEVEL

        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
        priority_hit = self._detect_new_level_priority(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
            force=True,
        )
        if priority_hit:
            source, confidence, x, y = priority_hit
            logger.info(
                "Priority override: %s detected at (%s, %s), transitioning immediately",
                source,
                x,
                y,
            )
            self._click_new_level_override(source=source)
            return State.TRANSITION_LEVEL

        return None

    def _click_new_level_override(self, source=None):
        now = time.monotonic()
        cooldown = getattr(config, "NEW_LEVEL_OVERRIDE_COOLDOWN", 0.0)
        if cooldown > 0 and now - self._last_new_level_override_time < cooldown:
            logger.debug("Priority override: skipping click sequence due to cooldown")
            return
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
        if source == "new level red icon":
            logger.debug("Priority override: red icon source, skipping transition position click")
            return
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

    def _capture(self, max_y=None, force=False):
        cache_key = max_y if max_y is not None else "full"
        cached = self._capture_cache.get(cache_key)
        now = time.monotonic()
        if not force and cached and now - cached[0] <= self._capture_cache_ttl:
            return cached[1]

        with self._capture_lock:
            frame = self.window_capture.capture(max_y=max_y)
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
            if self._should_interrupt_for_new_level(max_y=config.MAX_SEARCH_Y, force=True):
                return True
            now = time.monotonic()
        return False

    def _sleep_with_interrupt(self, duration):
        if duration <= 0:
            return False
        return self._sleep_until(time.monotonic() + duration)

    def _detect_new_level(self, screenshot=None, max_y=None, force=False):
        target_max_y = max_y if max_y is not None else config.MAX_SEARCH_Y
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
        else:
            self.vision_optimizer.update_new_level_miss()
        self._new_level_cache = {"timestamp": now, "result": result, "max_y": target_max_y}
        return result

    def _detect_new_level_red_icon(self, screenshot=None, max_y=None, force=False):
        target_max_y = max_y if max_y is not None else config.MAX_SEARCH_Y
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
        threshold = (
            self.vision_optimizer.new_level_red_icon_threshold
            if self.vision_optimizer.enabled
            else config.NEW_LEVEL_RED_ICON_THRESHOLD
        )

        for template_name, template, mask in self.available_red_icon_templates:
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

        min_matches = config.NEW_LEVEL_RED_ICON_MIN_MATCHES
        best_match = None
        for (x, y), matches in detections.items():
            if len(matches) >= min_matches:
                max_conf = max(conf for _, conf in matches)
                if best_match is None or max_conf > best_match[1]:
                    best_match = (True, max_conf, x, y)

        result = best_match or (False, 0.0, 0, 0)
        if result[0]:
            self.vision_optimizer.update_new_level_red_icon_confidence(result[1])
        else:
            self.vision_optimizer.update_new_level_red_icon_miss()

        self._new_level_red_icon_cache = {"timestamp": now, "result": result, "max_y": target_max_y}
        return result

    def _detect_new_level_priority(self, screenshot=None, max_y=None, force=False):
        red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
        )
        if red_found:
            self._mark_restaurant_completed("new level red icon", red_conf)
            return "new level red icon", red_conf, red_x, red_y

        found, confidence, x, y = self._detect_new_level(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
        )
        if found:
            self._mark_restaurant_completed("new level button", confidence)
            return "new level button", confidence, x, y

        return None

    def _should_interrupt_for_new_level(self, screenshot=None, max_y=None, force=False):
        priority_hit = self._detect_new_level_priority(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
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

        for template_name in self.red_icon_templates:
            if template_name not in self.templates:
                continue

            template, mask = self.templates[template_name]
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

        for template_name, template, mask in self.available_red_icon_templates:
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

    def _detect_red_icons_in_view(self, screenshot, max_y=None):
        if not self.available_red_icon_templates:
            return []

        detections = {}
        buckets = {}
        if max_y is not None:
            screenshot = screenshot[:max_y, :]
        threshold = (
            self.vision_optimizer.red_icon_threshold
            if self.vision_optimizer.enabled
            else config.RED_ICON_THRESHOLD
        )

        for template_name, template, mask in self.available_red_icon_templates:
            icons = self.image_matcher.find_all_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=80,
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

        min_matches = config.RED_ICON_MIN_MATCHES
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

        for template_name, template, mask in self.available_red_icon_templates:
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
        forbidden_zone_count = 0
        forbidden_zones = self.forbidden_zones

        for conf, x, y in red_icons:
            in_forbidden = any(
                zone_x_min <= x <= zone_x_max and zone_y_min <= y <= zone_y_max
                for zone_x_min, zone_x_max, zone_y_min, zone_y_max in forbidden_zones
            )

            if in_forbidden:
                forbidden_zone_count += 1
            else:
                filtered_icons.append((conf, x, y))

        return filtered_icons, forbidden_zone_count

    def _prioritize_red_icons(self, red_icons):
        def get_priority(icon):
            conf, x, y = icon
            for success_y in self.successful_red_icon_positions:
                if abs(y - success_y) < 50:
                    return (0, y)
            return (1, y)

        red_icons.sort(key=get_priority)
        return red_icons

    def _scroll_and_scan_for_red_icons(self, direction, scroll_duration):
        if direction == "up":
            logger.info("⬆ Scroll UP (scan)")
            self.mouse_controller.drag(
                config.SCROLL_END_POS[0],
                config.SCROLL_END_POS[1],
                config.SCROLL_START_POS[0],
                config.SCROLL_START_POS[1],
                duration=scroll_duration,
                relative=True,
            )
        else:
            logger.info("⬇ Scroll DOWN (scan)")
            self.mouse_controller.drag(
                config.SCROLL_START_POS[0],
                config.SCROLL_START_POS[1],
                config.SCROLL_END_POS[0],
                config.SCROLL_END_POS[1],
                duration=scroll_duration,
                relative=True,
            )

        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)

        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)

        if self._should_interrupt_for_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        ):
            return State.TRANSITION_LEVEL

        red_icons = self._detect_red_icons_in_view(limited_screenshot, max_y=config.MAX_SEARCH_Y)
        self.vision_optimizer.update_red_icon_scan([conf for conf, _, _ in red_icons])
        if not red_icons:
            return None

        filtered_icons, forbidden_zone_count = self._filter_forbidden_red_icons(red_icons)
        if forbidden_zone_count > 0:
            logger.info(f"Forbidden Zone Filter: {forbidden_zone_count} icons removed during scroll")

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

    def _build_available_red_icon_templates(self):
        available = []
        for template_name in self.red_icon_templates:
            if template_name in self.templates:
                template, mask = self.templates[template_name]
                available.append((template_name, template, mask))
        return available

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

        if self._should_interrupt_for_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
            force=True,
        ):
            logger.info("New level detected during scan, transitioning")
            return State.TRANSITION_LEVEL

        self.red_icons = self._detect_red_icons_in_view(
            screenshot,
            max_y=config.MAX_SEARCH_Y,
        )

        self.vision_optimizer.update_red_icon_scan([conf for conf, _, _ in self.red_icons])

        logger.info(
            "Red Icon Detection: %s valid icons found (min %s template matches)",
            len(self.red_icons),
            config.RED_ICON_MIN_MATCHES,
        )

        red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
            screenshot=screenshot,
            max_y=config.MAX_SEARCH_Y,
            force=True,
        )
        if red_found:
            self._mark_restaurant_completed("new level red icon", red_conf)
            logger.info(f"New level detected! Red icon at ({red_x}, {red_y})")
            return State.CHECK_NEW_LEVEL
        
        if not self.red_icons:
            logger.info("No valid red icons after scan; scrolling to search")
            self.no_red_icons_found = True
            return State.SCROLL
        else:
            filtered_icons, forbidden_zone_count = self._filter_forbidden_red_icons(self.red_icons)
            if forbidden_zone_count > 0:
                logger.info(f"Forbidden Zone Filter: {forbidden_zone_count} icons removed")
            
            if not filtered_icons:
                logger.info("No valid red icons after filtering; scrolling to search")
                self.no_red_icons_found = True
                return State.SCROLL
            
            self.red_icons = self._prioritize_red_icons(filtered_icons)
            
            logger.info(f"✓ {len(self.red_icons)} red icons ready to process")
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
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)
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
        base_threshold = (
            self.vision_optimizer.upgrade_station_threshold
            if self.vision_optimizer.enabled
            else config.UPGRADE_STATION_THRESHOLD
        )
        relaxed_threshold = base_threshold - 0.05
        retry_delay = self.tuner.search_interval
        
        for attempt in range(max_attempts):
            limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
            
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
        return State.OPEN_BOXES
    
    def handle_hold_upgrade_station(self, current_state):
        base_pos = self.upgrade_station_pos
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)
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
            return State.OPEN_BOXES
        
        logger.info("Holding upgrade station click...")

        max_hold_time = config.UPGRADE_HOLD_DURATION
        check_interval = config.UPGRADE_CHECK_INTERVAL
        start_time = time.monotonic()
        end_time = start_time + max_hold_time
        upgrade_missing_logged = False
        next_check_time = start_time + check_interval
        interrupt_state = None

        if not self.mouse_controller.mouse_down(x, y, relative=True):
            self.red_icon_processed_count += 1
            return State.OPEN_BOXES

        try:
            while True:
                now = time.monotonic()
                if now >= end_time:
                    break

                if now >= next_check_time:
                    limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)

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
                        screenshot=limited_screenshot,
                        max_y=config.MAX_SEARCH_Y,
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
        
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL
        
        self.red_icon_processed_count += 1
        
        logger.info("✓ Upgrade station complete → Stats upgrade next")
        return interrupt_state or State.UPGRADE_STATS
    
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
        
        scroll_duration = config.NO_ICON_SCROLL_DURATION if self.no_red_icons_found else config.SCROLL_DURATION

        if self.no_red_icons_found:
            logger.info("No red icons found → running up/down scan scroll sequence")
            for direction, count in (
                ("up", config.NO_ICON_SCROLL_UP_COUNT),
                ("down", config.NO_ICON_SCROLL_DOWN_COUNT),
            ):
                for _ in range(count):
                    state = self._scroll_and_scan_for_red_icons(direction, scroll_duration)
                    if state is not None:
                        return state
            return State.FIND_RED_ICONS

        if self.scroll_direction == 'up':
            logger.info(f"⬆ Scroll UP ({self.scroll_count + 1}/{self.max_scroll_count})")
            self.mouse_controller.drag(
                config.SCROLL_END_POS[0], config.SCROLL_END_POS[1],
                config.SCROLL_START_POS[0], config.SCROLL_START_POS[1],
                duration=scroll_duration, relative=True
            )
        else:  # down
            logger.info(f"⬇ Scroll DOWN ({self.scroll_count + 1}/{self.max_scroll_count})")
            self.mouse_controller.drag(
                config.SCROLL_START_POS[0], config.SCROLL_START_POS[1],
                config.SCROLL_END_POS[0], config.SCROLL_END_POS[1],
                duration=scroll_duration, relative=True
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
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL

        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
        if self._should_interrupt_for_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        ):
            return State.TRANSITION_LEVEL
        
        logger.info("Clicking new level button position")
        self.mouse_controller.click(config.NEW_LEVEL_BUTTON_POS[0], config.NEW_LEVEL_BUTTON_POS[1], relative=True)
        if config.NEW_LEVEL_BUTTON_DELAY > 0:
            if self._sleep_with_interrupt(config.NEW_LEVEL_BUTTON_DELAY):
                return State.TRANSITION_LEVEL
        
        logger.info("Triggering follow-up click after new level check")
        self.mouse_controller.click(166, 526, relative=True)
        if config.NEW_LEVEL_FOLLOWUP_DELAY > 0:
            if self._sleep_with_interrupt(config.NEW_LEVEL_FOLLOWUP_DELAY):
                return State.TRANSITION_LEVEL

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
                self._mark_restaurant_completed("new level button", confidence)
                logger.info(f"New level button found at ({x}, {y}) (attempt {attempt + 1})")
                self.mouse_controller.click(x, y, relative=True)
                if config.TRANSITION_POST_CLICK_DELAY > 0:
                    if self._sleep_with_interrupt(config.TRANSITION_POST_CLICK_DELAY):
                        return State.TRANSITION_LEVEL

                self.total_levels_completed += 1

                time_spent = 0
                if self.current_level_start_time:
                    completion_time = self.completion_detected_time or datetime.now()
                    time_spent = (completion_time - self.current_level_start_time).total_seconds()

                self.current_level_start_time = datetime.now()
                self.completion_detected_time = None
                self.completion_detected_by = None

                self.telegram.notify_new_level(self.total_levels_completed, time_spent)

                logger.info(f"Level {self.total_levels_completed} completed. Time spent: {time_spent:.1f}s")
                logger.info("Waiting for unlock button after level transition")
                return State.WAIT_FOR_UNLOCK
            
            if attempt < max_attempts - 1:
                if config.TRANSITION_RETRY_DELAY > 0:
                    if self._sleep_with_interrupt(config.TRANSITION_RETRY_DELAY):
                        return State.TRANSITION_LEVEL
        
        logger.warning("New level button not found after 5 attempts")
        self.scroll_direction = 'down'
        self.scroll_count = 0
        return State.FIND_RED_ICONS
    
    def handle_wait_for_unlock(self, current_state):
        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
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
        self._new_level_monitor_stop.set()
        if self._new_level_monitor_thread and self._new_level_monitor_thread.is_alive():
            self._new_level_monitor_thread.join(timeout=1.0)
        if self.overlay:
            self.overlay.stop()
        logger.info("Bot stopped")
