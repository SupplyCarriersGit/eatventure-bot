import json
import os
import time
import logging
import threading
from datetime import datetime

from window_capture import WindowCapture
from image_matcher import ImageMatcher
from mouse_controller import MouseController
from state_machine import StateMachine, State
from telegram_notifier import TelegramNotifier
from asset_scanner import AssetScanner
from search_utils import OscillatingSearcher
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
        if self.click_success_rate < config.ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD:
            self.click_delay = min(self.click_delay + config.ADAPTIVE_TUNER_CLICK_DELAY_STEP, config.ADAPTIVE_TUNER_MAX_CLICK_DELAY)
            self.move_delay = min(self.move_delay + config.ADAPTIVE_TUNER_MOVE_DELAY_STEP, config.ADAPTIVE_TUNER_MAX_MOVE_DELAY)
        elif self.click_success_rate > config.ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD:
            self.click_delay = max(self.click_delay - config.ADAPTIVE_TUNER_CLICK_DECREMENT, config.ADAPTIVE_TUNER_MIN_CLICK_DELAY)
            self.move_delay = max(self.move_delay - config.ADAPTIVE_TUNER_MOVE_DECREMENT, config.ADAPTIVE_TUNER_MIN_MOVE_DELAY)

    def _adjust_search_timing(self):
        if self.search_success_rate < config.ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD:
            self.search_interval = min(self.search_interval + config.ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP, config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL)
            self.upgrade_click_interval = min(
                self.upgrade_click_interval + config.ADAPTIVE_TUNER_UPGRADE_INTERVAL_STEP,
                config.ADAPTIVE_TUNER_MAX_UPGRADE_INTERVAL,
            )
        elif self.search_success_rate > config.ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD:
            self.search_interval = max(self.search_interval - config.ADAPTIVE_TUNER_SEARCH_DECREMENT, config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL)
            self.upgrade_click_interval = max(
                self.upgrade_click_interval - config.ADAPTIVE_TUNER_UPGRADE_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_UPGRADE_INTERVAL,
            )

    def reset(self):
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = config.CLICK_DELAY
        self.move_delay = config.MOUSE_MOVE_DELAY
        self.upgrade_click_interval = config.UPGRADE_CLICK_INTERVAL
        self.search_interval = config.UPGRADE_SEARCH_INTERVAL
        logger.info("AdaptiveTuner reset to defaults")


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
        self.box_threshold = config.BOX_THRESHOLD
        self.persistence = persistence
        self._red_icon_miss_count = 0
        self._new_level_miss_count = 0
        self._new_level_red_icon_miss_count = 0
        self._upgrade_station_miss_count = 0
        self._stats_upgrade_miss_count = 0
        self._box_miss_count = 0

    def _ema(self, current, new_value, alpha=None):
        blend = self.alpha if alpha is None else alpha
        return (1 - blend) * current + blend * new_value

    def _adaptive_alpha(self, confidence):
        if confidence <= 0:
            return self.alpha
        boost = max(0.0, min(1.0, (confidence - config.AI_VISION_CONFIDENCE_THRESHOLD))) * self.confidence_boost
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

    def update_box_confidence(self, confidence):
        if not self.enabled or confidence <= 0:
            return
        self._box_miss_count = 0
        target = max(
            config.AI_BOX_THRESHOLD_MIN,
            min(confidence, config.AI_BOX_THRESHOLD_MAX),
        )
        self.box_threshold = self._ema(
            self.box_threshold,
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

    def update_box_miss(self):
        if not self.enabled:
            return
        self._box_miss_count += 1
        if self._box_miss_count < config.AI_BOX_MISS_WINDOW:
            return
        self._box_miss_count = 0
        target = max(config.AI_BOX_THRESHOLD_MIN, self.box_threshold - config.AI_BOX_MISS_STEP)
        self.box_threshold = self._ema(self.box_threshold, target, self.alpha_max)
        self._persist()

    def reset(self):
        self.red_icon_threshold = config.RED_ICON_THRESHOLD
        self.new_level_threshold = config.NEW_LEVEL_THRESHOLD
        self.new_level_red_icon_threshold = config.NEW_LEVEL_RED_ICON_THRESHOLD
        self.upgrade_station_threshold = config.UPGRADE_STATION_THRESHOLD
        self.stats_upgrade_threshold = config.STATS_RED_ICON_THRESHOLD
        self.box_threshold = config.BOX_THRESHOLD
        self._red_icon_miss_count = 0
        self._new_level_miss_count = 0
        self._new_level_red_icon_miss_count = 0
        self._upgrade_station_miss_count = 0
        self._stats_upgrade_miss_count = 0
        self._box_miss_count = 0
        self._persist(force=True)
        logger.info("VisionOptimizer reset to defaults")

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
        if "box_threshold" in state:
            self.box_threshold = float(state["box_threshold"])

    def _persist(self, force=False):
        if not self.persistence:
            return
        state = {
            "red_icon_threshold": self.red_icon_threshold,
            "new_level_threshold": self.new_level_threshold,
            "new_level_red_icon_threshold": self.new_level_red_icon_threshold,
            "upgrade_station_threshold": self.upgrade_station_threshold,
            "stats_upgrade_threshold": self.stats_upgrade_threshold,
            "box_threshold": self.box_threshold,
        }
        self.persistence.save({key: float(value) for key, value in state.items()}, force=force)


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

    def save(self, state, force=False):
        if not self.path:
            return
        now = time.monotonic()
        if not force and self.save_interval > 0 and now - self._last_save_time < self.save_interval:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        self._last_save_time = now


class HistoricalLearner:
    def __init__(self, bot, persistence=None):
        self.bot = bot
        self.persistence = persistence
        self.enabled = config.AI_LEARNING_ENABLED
        self.interval = max(0.01, float(getattr(config, "AI_LEARNING_THREAD_INTERVAL", 0.05)))
        self.pair_window = max(2, int(getattr(config, "AI_LEARNING_PAIR_WINDOW", 2)))
        self.batch_window = max(2, int(getattr(config, "AI_LEARNING_BATCH_WINDOW", 7)))
        self.ema_alpha = max(0.01, min(0.8, float(getattr(config, "AI_LEARNING_EMA_ALPHA", 0.18))))
        self.auto_write_config = bool(getattr(config, "AI_LEARNING_AUTOWRITE_CONFIG", False))
        self.top_k = max(1, int(getattr(config, "AI_LEARNING_PROFILE_BLEND_TOP_K", 3)))
        self.min_improvement_ratio = max(0.0, float(getattr(config, "AI_LEARNING_MIN_IMPROVEMENT_RATIO", 0.03)))
        self.apply_cooldown = max(0.0, float(getattr(config, "AI_LEARNING_APPLY_COOLDOWN", 1.2)))
        self._last_apply_time = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._records = []
        self._total_completions = 0
        self._last_pair_processed = 0
        self._last_batch_processed = 0

        persisted = self.persistence.load() if self.persistence else {}
        if persisted:
            self._records = list(persisted.get("records", []))[-100:]
            self._total_completions = int(persisted.get("total_completions", len(self._records)))
            self._last_pair_processed = int(persisted.get("last_pair_processed", 0))
            self._last_batch_processed = int(persisted.get("last_batch_processed", 0))
            self._tuned_behavior = persisted.get("tuned_behavior", {})

            if self._tuned_behavior:
                logger.info("Historical learner applying persisted behavior profile")
                self.bot.apply_learned_behavior(self._tuned_behavior, reason="persisted")

            max_pair_processed = self._total_completions // self.pair_window
            max_batch_processed = self._total_completions // self.batch_window
            self._last_pair_processed = min(self._last_pair_processed, max_pair_processed)
            self._last_batch_processed = min(self._last_batch_processed, max_batch_processed)
        else:
            self._tuned_behavior = {}

    def start(self):
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="historical_learner", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=config.AI_LEARNING_THREAD_JOIN_TIMEOUT)
        self._persist()

    def record_completion(self, time_spent, source):
        if not self.enabled or time_spent <= 0:
            return
        snapshot = self.bot.get_runtime_behavior_snapshot()
        record = {
            "timestamp": time.time(),
            "time_spent": float(time_spent),
            "source": source,
            "behavior": snapshot,
        }
        with self._lock:
            self._records.append(record)
            self._records = self._records[-config.AI_LEARNING_RECORDS_LIMIT:]
            self._total_completions += 1
        self._persist()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._run_learning_cycle()
            except Exception:
                logger.exception("Historical learner cycle failed; continuing")
            time.sleep(max(0.01, self.interval))

    def _run_learning_cycle(self):
        with self._lock:
            records = list(self._records)
            total_completions = int(self._total_completions)

        if self._is_apply_cooldown_active():
            self._persist()
            return

        if total_completions >= self.pair_window and total_completions // self.pair_window > self._last_pair_processed:
            pair_records = records[-self.pair_window:]
            profile = self._build_profile(pair_records)
            self._apply_profile_if_improved(profile, pair_records, f"pair-{self.pair_window}")
            self._last_pair_processed = total_completions // self.pair_window

        # Re-check cooldown after pair application to prevent back-to-back
        # profile rewrites within the same learning pass.
        if self._is_apply_cooldown_active():
            self._persist()
            return

        if total_completions >= self.batch_window and total_completions // self.batch_window > self._last_batch_processed:
            batch_records = records[-self.batch_window:]
            profile = self._build_profile(batch_records)
            self._apply_profile_if_improved(profile, batch_records, f"batch-{self.batch_window}")
            self._last_batch_processed = total_completions // self.batch_window

        self._persist()

    def _is_apply_cooldown_active(self):
        if self.apply_cooldown <= 0:
            return False
        return (time.monotonic() - self._last_apply_time) < self.apply_cooldown

    def _build_profile(self, records):
        valid = [r for r in records if r.get("time_spent", 0) > 0 and r.get("behavior")]
        if not valid:
            return None
        ranked = sorted(valid, key=lambda item: item.get("time_spent", float("inf")))
        top = ranked[: self.top_k]
        profile = {"click_delay": 0.0, "move_delay": 0.0, "upgrade_click_interval": 0.0, "search_interval": 0.0}
        for record in top:
            behavior = record.get("behavior") or {}
            for key in profile:
                profile[key] += float(behavior.get(key, 0.0))
        count = float(len(top))
        return {key: value / count for key, value in profile.items()}

    def _apply_profile_if_improved(self, profile, records, label):
        if not profile or not records:
            return
        durations = [r.get("time_spent", 0.0) for r in records if r.get("time_spent", 0.0) > 0]
        if not durations:
            return
        best_time = min(durations)
        avg_time = sum(durations) / len(durations)
        if avg_time <= 0:
            return
        improvement_ratio = (avg_time - best_time) / avg_time
        if improvement_ratio < self.min_improvement_ratio:
            logger.debug(
                "Historical learner (%s) skipped profile apply; improvement %.2f%% below %.2f%%",
                label,
                improvement_ratio * 100,
                self.min_improvement_ratio * 100,
            )
            return
        self._apply_best_record({"behavior": profile, "time_spent": best_time}, label)
        self._last_apply_time = time.monotonic()

    def _ema(self, current, target):
        return (1 - self.ema_alpha) * current + self.ema_alpha * target

    def _clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def _apply_best_record(self, record, label):
        behavior = record.get("behavior") or {}
        if not behavior:
            return

        current = self.bot.get_runtime_behavior_snapshot()
        tuned = {}
        keys = ("click_delay", "move_delay", "upgrade_click_interval", "search_interval")
        for key in keys:
            if key not in behavior or key not in current:
                continue
            tuned[key] = self._ema(float(current[key]), float(behavior[key]))

        tuned["click_delay"] = self._clamp(
            tuned.get("click_delay", current["click_delay"]),
            config.AI_LEARNING_MIN_CLICK_DELAY,
            config.AI_LEARNING_MAX_CLICK_DELAY,
        )
        tuned["move_delay"] = self._clamp(
            tuned.get("move_delay", current["move_delay"]),
            config.AI_LEARNING_MIN_MOVE_DELAY,
            config.AI_LEARNING_MAX_MOVE_DELAY,
        )
        tuned["upgrade_click_interval"] = self._clamp(
            tuned.get("upgrade_click_interval", current["upgrade_click_interval"]),
            config.AI_LEARNING_MIN_UPGRADE_INTERVAL,
            config.AI_LEARNING_MAX_UPGRADE_INTERVAL,
        )
        tuned["search_interval"] = self._clamp(
            tuned.get("search_interval", current["search_interval"]),
            config.AI_LEARNING_MIN_SEARCH_INTERVAL,
            config.AI_LEARNING_MAX_SEARCH_INTERVAL,
        )

        self._tuned_behavior = tuned
        self.bot.apply_learned_behavior(tuned, reason=label, best_time=record.get("time_spent", 0.0))

    def _persist(self, force=False):
        if not self.persistence:
            return
        state = {
            "records": self._records[-config.AI_LEARNING_RECORDS_LIMIT:],
            "total_completions": self._total_completions,
            "last_pair_processed": self._last_pair_processed,
            "last_batch_processed": self._last_batch_processed,
            "tuned_behavior": self._tuned_behavior,
        }
        self.persistence.save(state, force=force)

    def reset(self):
        with self._lock:
            self._records = []
            self._total_completions = 0
            self._last_pair_processed = 0
            self._last_batch_processed = 0
            self._tuned_behavior = {}
        self._persist(force=True)
        logger.info("HistoricalLearner reset to defaults")


class EatventureBot:
    def __init__(self):
        logger.info("Initializing Eatventure Bot...")
        
        self.window_capture = WindowCapture(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.image_matcher = ImageMatcher(config.MATCH_THRESHOLD)
        self.mouse_controller = MouseController(
            self.window_capture.hwnd,
            config.CLICK_DELAY,
            config.MOUSE_MOVE_DELAY
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
        self.max_wait_for_unlock_attempts = getattr(config, "WAIT_FOR_UNLOCK_MAX_ATTEMPTS", 50)
        
        # Legacy directional scroll state removed.
        # One-Scroll Rule: execute_oscillating_search() is the only scroll driver.
        self.work_done = False
        self.cycle_counter = 0
        self.red_icon_processed_count = 0
        self.forbidden_icon_scrolls = 0
        self.scroll_offset_units = 0  # Tracks vertical drift from center
        
        self.successful_red_icon_positions = []
        self.upgrade_found_in_cycle = False
        self.consecutive_failed_cycles = 0
        
        self.total_levels_completed = 0
        self._last_transition_time = 0.0
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
        self.searcher = OscillatingSearcher(self)
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
        self._last_new_level_fail_time = 0.0
        self._last_idle_click_time = 0.0
        self._state_last_run_at = {}
        self._recent_red_icon_history = []
        self._no_red_scroll_cycle_pending = False

        self.forbidden_zones = [
            (zone["x_min"], zone["x_max"], zone["y_min"], zone["y_max"])
            for zone in getattr(config, "FORBIDDEN_ZONES", [])
        ]

        self.overlay = None
        
        logger.info("Bot initialized successfully")

    def _record_new_level_interrupt(self, source, confidence, x, y):
        if self._should_ignore_new_level_signal(source=source):
            logger.debug(
                "Ignoring background %s signal while in %s",
                source,
                self.state_machine.get_state_name(),
            )
            return

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
        if interrupt and self._should_ignore_new_level_signal(source=interrupt.get("source")):
            return None
        return interrupt

    def _should_ignore_new_level_signal(self, source, state=None):
        # Ignore ALL new level signals (icons and buttons) during critical phases.
        # This prevents the bot from jumping back to TRANSITION_LEVEL while 
        # it is already waiting for the new level to settle.
        active_state = state or self.state_machine.get_state()
        critical_states = (
            State.WAIT_FOR_UNLOCK,
            State.TRANSITION_LEVEL,
            State.CHECK_NEW_LEVEL,
            State.CHECK_UNLOCK,
            State.SEARCH_UPGRADE_STATION,
            State.HOLD_UPGRADE_STATION,
        )
        if active_state in critical_states:
            return True
            
        # Also enforce a hard cooldown after a transition to handle game lag/echoes
        if source == "new level button" or source == "new level red icon":
            if time.monotonic() - self._last_transition_time < 30.0:
                return True
                
        return False

    def _monitor_new_level(self):
        interval = config.NEW_LEVEL_MONITOR_INTERVAL
        while not self._new_level_monitor_stop.is_set():
            if self._new_level_event.is_set():
                time.sleep(max(interval, 0.01))
                continue

            monitor_screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y, force=True)
            limited_screenshot = monitor_screenshot[:config.MAX_SEARCH_Y, :]

            red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
                screenshot=monitor_screenshot,
                max_y=config.EXTENDED_SEARCH_Y,
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
        self.mouse_controller.move_delay = self.tuner.move_delay

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

    def _scroll_away_from_forbidden_zone(self, y_position):
        # Disabled intentionally.
        #
        # One-Scroll Rule: all scrolling must be performed only by
        # execute_oscillating_search(). Forbidden-zone handling now skips the
        # blocked icon instead of performing an ad-hoc directional scroll.
        logger.info(
            "Red icon in forbidden zone at y=%s; skipping icon (ad-hoc forbidden-zone scrolling disabled)",
            y_position,
        )
        return False

    def resolve_priority_state(self, current_state):
        if current_state in (State.CHECK_NEW_LEVEL, State.TRANSITION_LEVEL, State.WAIT_FOR_UNLOCK):
            return None

        if current_state == State.FIND_RED_ICONS and self._no_red_scroll_cycle_pending:
            logger.info("Priority override: continuing no-red scroll cycle after fallback asset scan")
            self._no_red_scroll_cycle_pending = False
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
            self._click_new_level_override(source=interrupt["source"])
            return State.TRANSITION_LEVEL

        priority_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
        priority_hit = self._detect_new_level_priority(
            screenshot=priority_screenshot,
            max_y=config.EXTENDED_SEARCH_Y,
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
            if self._no_red_scroll_cycle_pending:
                logger.info("Clearing deferred no-red scroll due to immediate level transition")
                self._no_red_scroll_cycle_pending = False
            self._click_new_level_override(source=source)
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
        immediate_threshold = getattr(config, "RED_ICON_PIXEL_THRESHOLD", 50) * 1.5 # Super solid trigger
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
        for conf, x, y, px_count in red_icons:
            # Requirement: Pixel Density Trigger (Immediate success if high density)
            if px_count >= immediate_threshold:
                logger.debug(f"Immediate trigger: high pixel density ({px_count}) at ({x}, {y})")
                stable.append((conf, x, y))
                continue

            hits = 0
            best_conf = conf
            for entry in history:
                for h_conf, hx, hy, hpx in entry["icons"]:
                    if abs(hx - x) <= radius and abs(hy - y) <= radius:
                        hits += 1
                        if h_conf > best_conf:
                            best_conf = h_conf
                        break
            if hits >= min_hits:
                stable.append((best_conf, x, y))

        return stable

    def _click_new_level_override(self, source=None):
        now = time.monotonic()
        cooldown = getattr(config, "NEW_LEVEL_OVERRIDE_COOLDOWN", 0.0)
        if cooldown > 0 and now - self._last_new_level_override_time < cooldown:
            logger.debug("Priority override: skipping click sequence due to cooldown")
            return
        self._last_new_level_override_time = now

        self._mark_restaurant_completed(source or "priority override")

        logger.info(
            "Priority override: clicking new level button position at (%s, %s)",
            config.NEW_LEVEL_BUTTON_POS[0],
            config.NEW_LEVEL_BUTTON_POS[1],
        )
        self.mouse_controller.click(
            config.NEW_LEVEL_BUTTON_POS[0],
            config.NEW_LEVEL_BUTTON_POS[1],
            relative=True,
        )

        button_delay = getattr(config, "NEW_LEVEL_BUTTON_DELAY", 0.02)
        if button_delay > 0:
            time.sleep(button_delay)

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
        
        if button_delay > 0:
            time.sleep(button_delay)

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
            time.sleep(max(0, target_time - now))
            return False

        while now < target_time:
            remaining = max(0, target_time - now)
            time.sleep(min(interval, remaining))
            if self._new_level_event.is_set():
                interrupt = self._new_level_interrupt
                if interrupt and self._should_ignore_new_level_signal(source=interrupt.get("source")):
                    self._new_level_event.clear()
                    now = time.monotonic()
                    continue
                return True
            if self._should_interrupt_for_new_level(max_y=config.MAX_SEARCH_Y, force=True):
                return True
            now = time.monotonic()
        return False

    def _sleep_with_interrupt(self, duration):
        if duration <= 0:
            return False
        return self._sleep_until(time.monotonic() + duration)

    def _sleep_with_search_interrupt(self, duration):
        """
        Pauses for the specified duration but checks for Red Icons and Level Transitions.
        Returns a State if an interrupt is detected, otherwise None.
        """
        if duration <= 0:
            return None
            
        target_time = time.monotonic() + duration
        interval = max(0.01, config.NEW_LEVEL_INTERRUPT_INTERVAL)
        
        while time.monotonic() < target_time:
            # 1. Check for Level Transition (High Priority)
            if self._should_interrupt_for_new_level(force=True):
                return State.TRANSITION_LEVEL
                
            # 2. Check for Red Icons
            screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)
            red_icons = self._detect_red_icons_in_view(screenshot, max_y=config.MAX_SEARCH_Y)
            
            # Implementation: Immediate Trigger for high density
            immediate_threshold = getattr(config, "RED_ICON_PIXEL_THRESHOLD", 50) * 1.5
            
            if red_icons:
                filtered, _ = self._filter_forbidden_red_icons(red_icons)
                if filtered:
                    # Check if any pass the 'immediate' threshold
                    has_immediate = any(px >= immediate_threshold for *_, px in filtered)
                    
                    if has_immediate:
                        self.red_icons = self._prioritize_red_icons(filtered)
                        self.current_red_icon_index = 0
                        self.work_done = True
                        return State.CLICK_RED_ICON
                    
                    # Otherwise, use standard stability check (which requires 3+ hits)
                    stable = self._stable_red_icons(filtered)
                    if stable:
                        self.red_icons = self._prioritize_red_icons(stable)
                        self.current_red_icon_index = 0
                        self.work_done = True
                        return State.CLICK_RED_ICON
            
            # 3. Check for Fallback Assets (Upgrade Station, Boxes)
            clicked = self._scan_and_click_non_red_assets(screenshot)
            if clicked > 0:
                # We clicked something, need to re-evaluate state
                return State.FIND_RED_ICONS

            time.sleep(min(interval, max(0, target_time - time.monotonic())))
            
        return None

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
        now = time.monotonic()
        
        # Check cooldown after a recent failure to prevent click loops on non-level red icons (e.g. Map rewards)
        fail_cooldown = config.NEW_LEVEL_FAIL_COOLDOWN
        if now - self._last_new_level_fail_time < fail_cooldown:
            return (False, 0.0, 0, 0)

        target_max_y = max_y if max_y is not None else config.MAX_SEARCH_Y
        cached = self._new_level_red_icon_cache
        cache_ttl = config.NEW_LEVEL_RED_ICON_CACHE_TTL
        if not force and cached["max_y"] == target_max_y and now - cached["timestamp"] <= cache_ttl:
            return cached["result"]

        max_template_width = 0
        max_template_height = 0
        for _, template, _ in self._iter_red_icon_templates():
            max_template_height = max(max_template_height, int(template.shape[0]))
            max_template_width = max(max_template_width, int(template.shape[1]))

        roi_pad_x = max(2, max_template_width // 2)
        roi_pad_y = max(2, max_template_height // 2)

        # The new-level red icon is configured near the bottom of the screen.
        # If callers provide a cropped frame (e.g. MAX_SEARCH_Y), the ROI can
        # be clipped out entirely and produce guaranteed false negatives.
        required_bottom = config.NEW_LEVEL_RED_ICON_Y_MAX + roi_pad_y
        if screenshot is None:
            screenshot = self._capture(max_y=target_max_y, force=force)

        if screenshot.shape[0] < required_bottom and max_y is None:
            recapture_max_y = max(target_max_y, required_bottom)
            screenshot = self._capture(max_y=recapture_max_y, force=force)
            target_max_y = recapture_max_y

        height, width = screenshot.shape[:2]
        x_min = max(0, config.NEW_LEVEL_RED_ICON_X_MIN - roi_pad_x)
        x_max = min(width, config.NEW_LEVEL_RED_ICON_X_MAX + roi_pad_x)
        y_min = max(0, config.NEW_LEVEL_RED_ICON_Y_MIN - roi_pad_y)
        y_max = min(height, config.NEW_LEVEL_RED_ICON_Y_MAX + roi_pad_y)

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
                min_distance=config.RED_ICON_MIN_DISTANCE,
                template_name=template_name,
            )
            for conf, x, y in icons:
                abs_x = x + x_min
                abs_y = y + y_min
                passed_color_gate, _ = self._passes_red_color_gate(screenshot, abs_x, abs_y)
                if not passed_color_gate:
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
                max_conf = max(conf for _, conf, _ in matches)
                if best_match is None or max_conf > best_match[1]:
                    best_match = (True, max_conf, x, y)

        self._update_red_template_priority(template_hits)
        result = best_match or (False, 0.0, 0, 0)
        if result[0]:
            self.vision_optimizer.update_new_level_red_icon_confidence(result[1])
        else:
            self.vision_optimizer.update_new_level_red_icon_miss()

        self._new_level_red_icon_cache = {"timestamp": now, "result": result, "max_y": target_max_y}
        return result

    def _detect_new_level_priority(self, screenshot=None, max_y=None, force=False):
        found, confidence, x, y = self._detect_new_level(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
        )
        if found:
            self._mark_restaurant_completed("new level button", confidence)
            return "new level button", confidence, x, y

        red_found, red_conf, red_x, red_y = self._detect_new_level_red_icon(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
        )
        if red_found:
            self._mark_restaurant_completed("new level red icon", red_conf)
            return "new level red icon", red_conf, red_x, red_y

        return None

    def _should_interrupt_for_new_level(self, screenshot=None, max_y=None, force=False):
        priority_hit = self._detect_new_level_priority(
            screenshot=screenshot,
            max_y=max_y,
            force=force,
        )
        if priority_hit:
            source, confidence, x, y = priority_hit
            
            # During critical station interaction phases, only interrupt if we see the actual 
            # renovation button, never just the red icon on the map which could be a reward.
            if self._should_ignore_new_level_signal(source=source):
                return False

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
        template_hits = {}

        for template_name, template, mask in self._iter_red_icon_templates():
            icons = self.image_matcher.find_all_templates(
                roi,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=config.RED_ICON_MIN_DISTANCE,
                template_name=template_name,
            )

            if icons:
                for conf, x, y in icons:
                    abs_x = x + x_min
                    abs_y = y + y_min
                    passed_color_gate, _ = self._passes_red_color_gate(screenshot, abs_x, abs_y)
                    if not passed_color_gate:
                        continue
                    best_confidence = max(best_confidence, conf)
                    template_hits[template_name] = template_hits.get(template_name, 0) + 1

        self._update_red_template_priority(template_hits)
        return best_confidence > 0, best_confidence

    def _merge_detection(self, detections, buckets, x, y, template_name, conf, proximity=None, bucket_size=None, pixel_count=0):
        prox = proximity if proximity is not None else config.RED_ICON_MERGE_PROXIMITY
        bsize = bucket_size if bucket_size is not None else config.RED_ICON_MERGE_BUCKET_SIZE
        bucket_x = x // bsize
        bucket_y = y // bsize
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in buckets.get((bucket_x + dx, bucket_y + dy), []):
                    if abs(x - px) < prox and abs(y - py) < prox:
                        detections[(px, py)].append((template_name, conf, pixel_count))
                        return

        detections[(x, y)] = [(template_name, conf, pixel_count)]
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
            passed_color_gate, _ = self._passes_red_color_gate(screenshot, abs_x, abs_y)
            if not passed_color_gate:
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

    def _detect_red_icons_in_view(self, screenshot, max_y=None, min_distance=80, threshold_override=None, min_matches_override=None, relaxed_color=False):
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
                passed, pixel_count = self._passes_red_color_gate(screenshot, x, y, relaxed=relaxed_color)
                if not passed:
                    continue
                self._merge_detection(
                    detections,
                    buckets,
                    x,
                    y,
                    template_name,
                    conf,
                    pixel_count=pixel_count
                )
                template_hits[template_name] = template_hits.get(template_name, 0) + 1

        self._update_red_template_priority(template_hits)

        min_matches = config.RED_ICON_MIN_MATCHES if min_matches_override is None else min_matches_override
        red_icons = []
        for (x, y), matches in detections.items():
            if len(matches) >= min_matches:
                max_conf = max(conf for _, conf, _ in matches)
                max_pixel_count = max(px for _, _, px in matches)
                red_icons.append((max_conf, x, y, max_pixel_count))
        return red_icons

    def _is_red_icon_present_at(self, x, y, screenshot=None, threshold_override=None):
        if not self.available_red_icon_templates:
            return False

        target_screenshot = screenshot if screenshot is not None else self._capture(max_y=config.MAX_SEARCH_Y)

        if config.RED_ICON_COLOR_CHECK:
            show_mask = getattr(config, "DEBUG_VISION", False)
            pixel_count = self.image_matcher.count_red_pixels(
                target_screenshot, x, y,
                size=getattr(config, "RED_ICON_COLOR_SAMPLE_SIZE", 24),
                show_mask=show_mask
            )
            if pixel_count < getattr(config, "RED_ICON_PIXEL_THRESHOLD", 50):
                return False

        if threshold_override is not None:
            threshold = threshold_override
        else:
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

    def _passes_red_color_gate(self, screenshot, x, y, relaxed=False):
        """
        Requirement: Pixel Density Trigger.
        Counts red pixels in ROI after dilation.
        Returns: (passed, pixel_count)
        """
        show_mask = getattr(config, "DEBUG_VISION", False)
        pixel_count = self.image_matcher.count_red_pixels(
            screenshot, x, y, 
            size=getattr(config, "RED_ICON_COLOR_SAMPLE_SIZE", 24),
            show_mask=show_mask
        )
        
        threshold = getattr(config, "RED_ICON_PIXEL_THRESHOLD", 50)
        if relaxed:
            threshold = int(threshold * 0.7)
            
        return pixel_count >= threshold, pixel_count

    def _filter_forbidden_red_icons(self, red_icons):
        filtered_icons = []
        forbidden_zone_count = 0
        forbidden_zones = self.forbidden_zones

        for icon in red_icons:
            conf, x, y = icon[:3]
            in_forbidden = any(
                zone_x_min <= x <= zone_x_max and zone_y_min <= y <= zone_y_max
                for zone_x_min, zone_x_max, zone_y_min, zone_y_max in forbidden_zones
            )

            if in_forbidden:
                forbidden_zone_count += 1
            else:
                filtered_icons.append(icon)

        return filtered_icons, forbidden_zone_count

    def _prioritize_red_icons(self, red_icons):
        def get_priority(icon):
            conf, x, y = icon[:3]
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

    def check_priority_targets(self):
        """STEP A: Priority Scan. Checks for Red Icons and Level Transitions."""
        screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)
        
        # 1. Check for Level Transition
        if self._should_interrupt_for_new_level(screenshot=screenshot, force=True):
            return State.TRANSITION_LEVEL

        # 2. Check for Red Icons
        red_icons = self._detect_red_icons_in_view(screenshot, max_y=config.MAX_SEARCH_Y)
        # Apply Temporal Consistency Check (Debouncing)
        red_icons = self._stable_red_icons(red_icons)
        
        if red_icons:
            filtered, _ = self._filter_forbidden_red_icons(red_icons)
            if filtered:
                self.red_icons = self._prioritize_red_icons(filtered)
                self.current_red_icon_index = 0
                self.red_icon_cycle_count = 0
                self.work_done = True
                # Return 'RED_ICON_FOUND' status (State.CLICK_RED_ICON)
                return State.CLICK_RED_ICON
        return None

    def check_main_success(self):
        """STEP B: Main Target Scan. Reserved for specific success conditions."""
        # In current context, Red Icons are the primary success, handled by priority.
        return None

    def check_fallbacks(self):
        """STEP C: Fallback Scan. Clicks boxes and stations without returning success."""
        screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)
        self._scan_and_click_non_red_assets(screenshot)

    def execute_oscillating_search(self):
        """
        Principal Architect Refactor: Uses the new OscillatingSearcher class 
        to execute a strict 5-step Scan-First protocol.
        """
        # Execute cycle with 3 distinct callback layers
        target_state = self.searcher.execute_cycle(
            check_priority=self.check_priority_targets,
            check_main_target=self.check_main_success,
            check_fallbacks=self.check_fallbacks
        )
        
        if target_state:
            # Cycle cooldown if found
            cooldown = getattr(config, "OSCILLATION_CYCLE_COOLDOWN", 0)
            if cooldown > 0:
                time.sleep(cooldown)
            return target_state
        
        # Exhausted retries -> return to base scanning
        return State.FIND_RED_ICONS


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
            found, confidence, x, y = self.image_matcher.find_template(
                screenshot,
                template,
                mask=mask,
                threshold=box_threshold,
                template_name=f"{box_name}-no-red-fallback",
            )

            if not found:
                self.vision_optimizer.update_box_miss()
                continue

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
    
    def wipe_memory(self):
        logger.info("Wiping AI memory...")
        self.tuner.reset()
        self.vision_optimizer.reset()
        self.historical_learner.reset()
        
        self._red_template_hit_counts = {}
        self._red_template_priority = []
        self._red_template_last_seen = {}
        self._recent_red_icon_history = []
        self._reset_search_cycle(reason="wipe_memory")
        
        # Apply the defaults back to mouse controller
        self._apply_tuning()
        
        logger.info("AI memory wiped successfully. Bot starting fresh.")
    
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
        self.red_icons = self._stable_red_icons(self.red_icons)

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
            # Verify with a template match to be sure it's not just a reward
            found_btn, _, _, _ = self._detect_new_level(screenshot=screenshot, max_y=config.MAX_SEARCH_Y)
            
            if self.red_icons and not found_btn:
                logger.info("Potential new level (red icon) detected on map, but prioritizing station upgrades first")
            else:
                self._mark_restaurant_completed("new level red icon", red_conf)
                logger.info(f"New level detected! Red icon at ({red_x}, {red_y})")
                return State.CHECK_NEW_LEVEL
        
        if not self.red_icons:
            clicked_targets = self._scan_and_click_non_red_assets(limited_screenshot)
            if clicked_targets > 0:
                logger.info(
                    "No red icons detected; fallback clicked %s non-red assets. Proceeding to scrolling cycle.",
                    clicked_targets,
                )
                return State.SCROLL

            logger.info("No valid red icons after scan; no fallback assets found, scrolling to search")
            return State.SCROLL
        else:
            filtered_icons, forbidden_zone_count = self._filter_forbidden_red_icons(self.red_icons)
            if forbidden_zone_count > 0:
                logger.info(f"Forbidden Zone Filter: {forbidden_zone_count} icons removed")
            
            if not filtered_icons:
                clicked_targets = self._scan_and_click_non_red_assets(limited_screenshot)
                if clicked_targets > 0:
                    logger.info(
                        "Filtered red icons to zero; fallback clicked %s non-red assets. Proceeding to scrolling cycle.",
                        clicked_targets,
                    )
                    return State.SCROLL

                logger.info("No valid red icons after filtering; no fallback assets found, scrolling to search")
                return State.SCROLL
            
            self.red_icons = self._prioritize_red_icons(filtered_icons)
            
            logger.info(f"✓ {len(self.red_icons)} red icons ready to process")
            self.current_red_icon_index = 0
            self.red_icon_cycle_count = 0
            # Removed: reset oscillation cycle here to allow persistence
            self.work_done = True
            return State.CLICK_RED_ICON
    
    def handle_click_red_icon(self, current_state):
        if self.current_red_icon_index >= len(self.red_icons):
            logger.info("All red icons processed, continuing cycle")
            return State.OPEN_BOXES
        
        confidence, x, y = self.red_icons[self.current_red_icon_index]
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y, force=True)
        
        # Calculate relaxed threshold for verification (matching search cycle logic)
        base_threshold = (
            self.vision_optimizer.red_icon_threshold
            if self.vision_optimizer.enabled
            else config.RED_ICON_THRESHOLD
        )
        relaxed_threshold = max(0.0, base_threshold - 0.04) # Match SCROLL_RED_ICON_THRESHOLD_DROP approx
        
        if not self._is_red_icon_present_at(x, y, screenshot=limited_screenshot, threshold_override=relaxed_threshold):
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
            
            if self._new_level_event.is_set():
                return State.TRANSITION_LEVEL
                
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
        
        clicked_unlock = False
        if "unlock" in self.templates:
            template, mask = self.templates["unlock"]
            found, confidence, x, y = self.image_matcher.find_template(
                limited_screenshot, template, mask=mask,
                threshold=config.UNLOCK_THRESHOLD, template_name="unlock"
            )
            
            if found:
                if self.mouse_controller.is_in_forbidden_zone(x, y):
                    logger.warning("Unlock button in forbidden zone, skipping")
                else:
                    logger.info("Unlock found, clicking")
                    clicked_unlock = self.mouse_controller.click(x, y, relative=True)

        if clicked_unlock:
            if self._sleep_with_interrupt(config.STATE_DELAY):
                return State.TRANSITION_LEVEL
            return self.handle_search_upgrade_station(current_state)

        return State.SEARCH_UPGRADE_STATION
    
    def handle_search_upgrade_station(self, current_state):
        max_attempts = config.UPGRADE_STATION_SEARCH_MAX_ATTEMPTS
        base_threshold = (
            self.vision_optimizer.upgrade_station_threshold
            if self.vision_optimizer.enabled
            else config.UPGRADE_STATION_THRESHOLD
        )
        relaxed_threshold = base_threshold - config.UPGRADE_STATION_RELAXED_THRESHOLD_DROP
        retry_delay = self.tuner.search_interval
        
        for attempt in range(max_attempts):
            limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
            
            if "upgradeStation" in self.templates:
                template, mask = self.templates["upgradeStation"]
                
                current_threshold = base_threshold if attempt < config.UPGRADE_STATION_RELAXED_ATTEMPT_TRIGGER else relaxed_threshold
                
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
            self.current_red_icon_index += 1
            if self.current_red_icon_index < len(self.red_icons):
                return State.CLICK_RED_ICON
            return State.OPEN_BOXES
        
        logger.info("Holding upgrade station click (Exempt from interrupts)...")

        # Define interrupt check for hold_at (Disabled for exemption)
        def check_interrupts():
            return False

        start_time = time.monotonic()
        
        # Use the hold action - interrupts are now disabled via the callback
        self.mouse_controller.hold_at(
            x, y, 
            duration=config.UPGRADE_HOLD_DURATION, 
            relative=True, 
            interrupt_check=check_interrupts
        )

        elapsed_time = time.monotonic() - start_time
        logger.info(f"Clicking complete: hold duration {elapsed_time:.1f}s")
        
        self._click_idle()
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL
        
        self.red_icon_processed_count += 1
        self.current_red_icon_index += 1

        logger.info("✓ Upgrade station complete → Stats upgrade next")
        return State.UPGRADE_STATS
    
    def handle_upgrade_stats(self, current_state):
        logger.info("⬆ Stats upgrade starting (Exempt from interrupts)")
        self._click_idle()
        
        extended_screenshot = self._capture(max_y=config.EXTENDED_SEARCH_Y)
        
        has_stats_icon, stats_confidence = self._has_stats_upgrade_icon(extended_screenshot)
        if not has_stats_icon:
            logger.info("✗ No stats icon, skipping")
            self.vision_optimizer.update_stats_upgrade_miss()
            return State.SCROLL

        self.vision_optimizer.update_stats_upgrade_confidence(stats_confidence)
        
        logger.info("✓ Stats icon found, upgrading")
        self.mouse_controller.click(config.STATS_UPGRADE_BUTTON_POS[0], config.STATS_UPGRADE_BUTTON_POS[1], relative=True)
        # Use standard non-interruptible sleep
        time.sleep(config.STATE_DELAY)
        
        start_time = time.monotonic()
        next_click_time = start_time
        while time.monotonic() - start_time < config.STATS_UPGRADE_CLICK_DURATION:
            self.mouse_controller.click(
                config.STATS_UPGRADE_POS[0],
                config.STATS_UPGRADE_POS[1],
                relative=True,
                wait_after=False,
            )
            
            # Use standard non-interruptible sleep
            sleep_duration = max(0, next_click_time + config.STATS_UPGRADE_CLICK_DELAY - time.monotonic())
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            
            next_click_time = max(
                next_click_time + config.STATS_UPGRADE_CLICK_DELAY,
                time.monotonic()
            )
        
        self._click_idle()
        logger.info("========== STAT UPGRADE COMPLETED ==========")
        return State.OPEN_BOXES
    
    def handle_open_boxes(self, current_state):
        self._click_idle()
        
        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

        if self._should_interrupt_for_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
            force=True,
        ):
            logger.info("New level found during box opening, transitioning")
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
                found, confidence, x, y = self.image_matcher.find_template(
                    limited_screenshot, template, mask=mask,
                    threshold=box_threshold, template_name=box_name
                )
                
                if found:
                    if self.mouse_controller.is_in_forbidden_zone(x, y):
                        logger.debug(f"{box_name} in forbidden zone, skipping")
                    else:
                        self.mouse_controller.click(x, y, relative=True)
                        boxes_found += 1
                        self.vision_optimizer.update_box_confidence(confidence)
                else:
                    self.vision_optimizer.update_box_miss()
        
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
        self._click_idle()
        
        # 1. DRIFT CORRECTION: If we are not at center (due to interrupt), return to center first.
        # This ensures the IOS pattern always starts from a known reference point.
        if abs(self.scroll_offset_units) > 0.01:
            logger.info(f"Drift detected ({self.scroll_offset_units:.2f} units). Correcting to center before search.")
            # If offset is positive (UP), we need to scroll DOWN (direction=1)
            # If offset is negative (DOWN), we need to scroll UP (direction=-1)
            direction_int = 1 if self.scroll_offset_units > 0 else -1
            correction_dist = abs(self.scroll_offset_units)
            
            # Perform correction using the new searcher's single source of truth
            self.searcher.perform_scroll(
                direction=direction_int,
                distance_ratio=correction_dist,
                duration=config.SCROLL_DURATION
            )
            return State.FIND_RED_ICONS

        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

        if self._should_interrupt_for_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
            force=True,
        ):
            logger.info("New level detected before scroll, transitioning")
            return State.TRANSITION_LEVEL
        
        # We now rely entirely on the Incremental Oscillating Search pattern
        # when the bot enters the SCROLL state (which happens when no icons are found).
        logger.info("Executing Incremental Oscillating Search")
        return self.execute_oscillating_search()
    
    def handle_check_new_level(self, current_state):
        self._click_idle()
        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL

        limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)
        if self._should_interrupt_for_new_level(
            screenshot=limited_screenshot,
            max_y=config.MAX_SEARCH_Y,
        ):
            return State.TRANSITION_LEVEL
        
        logger.info(f"Clicking new level button position at {config.NEW_LEVEL_BUTTON_POS}")
        self.mouse_controller.click(config.NEW_LEVEL_BUTTON_POS[0], config.NEW_LEVEL_BUTTON_POS[1], relative=True)

        button_delay = getattr(config, "NEW_LEVEL_BUTTON_DELAY", 0.02)
        if button_delay > 0:
            if self._sleep_with_interrupt(button_delay):
                return State.TRANSITION_LEVEL

        logger.info(f"Clicking new level position at {config.NEW_LEVEL_POS}")
        self.mouse_controller.click(config.NEW_LEVEL_POS[0], config.NEW_LEVEL_POS[1], relative=True)
        
        if button_delay > 0:
            if self._sleep_with_interrupt(button_delay):
                return State.TRANSITION_LEVEL
        
        logger.info(f"Clicking level transition position at {config.LEVEL_TRANSITION_POS}")
        self.mouse_controller.click(config.LEVEL_TRANSITION_POS[0], config.LEVEL_TRANSITION_POS[1], relative=True)
        
        followup_delay = getattr(config, "NEW_LEVEL_FOLLOWUP_DELAY", 0.02)
        if followup_delay > 0:
            if self._sleep_with_interrupt(followup_delay):
                return State.TRANSITION_LEVEL

        return State.TRANSITION_LEVEL
    
    def handle_transition_level(self, current_state):
        self._click_idle()
        
        max_attempts = config.LEVEL_TRANSITION_MAX_ATTEMPTS
        
        # Check if we already marked completion recently (e.g. via override)
        if self.completion_detected_time and (datetime.now() - self.completion_detected_time).total_seconds() < config.LEVEL_COMPLETION_RECENCY_WINDOW:
            logger.info("Completion already marked recently; proceeding to transition bookkeeping")
            return self._finalize_transition()

        for attempt in range(max_attempts):
            limited_screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

            found, confidence, x, y = self._detect_new_level(
                screenshot=limited_screenshot,
                max_y=config.MAX_SEARCH_Y,
            )
            if found:
                self._mark_restaurant_completed("new level button", confidence)
                logger.info(f"New level button found at ({x}, {y}); clicking config.NEW_LEVEL_BUTTON_POS at {config.NEW_LEVEL_BUTTON_POS}")
                
                # Use fixed config positions as requested
                self.mouse_controller.click(config.NEW_LEVEL_BUTTON_POS[0], config.NEW_LEVEL_BUTTON_POS[1], relative=True)

                button_delay = getattr(config, "NEW_LEVEL_BUTTON_DELAY", 0.02)
                if button_delay > 0:
                    time.sleep(button_delay)

                logger.info(f"Clicking new level position at {config.NEW_LEVEL_POS}")
                self.mouse_controller.click(config.NEW_LEVEL_POS[0], config.NEW_LEVEL_POS[1], relative=True)
                
                if button_delay > 0:
                    time.sleep(button_delay)

                logger.info(f"Clicking level transition position at {config.LEVEL_TRANSITION_POS}")
                self.mouse_controller.click(config.LEVEL_TRANSITION_POS[0], config.LEVEL_TRANSITION_POS[1], relative=True)

                if config.TRANSITION_POST_CLICK_DELAY > 0:
                    if self._sleep_with_interrupt(config.TRANSITION_POST_CLICK_DELAY):
                        return State.TRANSITION_LEVEL

                return self._finalize_transition()
            
            if attempt < max_attempts - 1:
                if config.TRANSITION_RETRY_DELAY > 0:
                    if self._sleep_with_interrupt(config.TRANSITION_RETRY_DELAY):
                        return State.TRANSITION_LEVEL
        
        logger.warning("New level button not found after 5 attempts")
        self._last_new_level_fail_time = time.monotonic()
        return State.FIND_RED_ICONS

    def _finalize_transition(self):
        self.total_levels_completed += 1
        self._last_transition_time = time.monotonic()

        time_spent = 0
        if self.current_level_start_time:
            completion_time = self.completion_detected_time or datetime.now()
            time_spent = (completion_time - self.current_level_start_time).total_seconds()

        completion_source = self.completion_detected_by or "new level button"
        self.current_level_start_time = datetime.now()
        self.completion_detected_time = None
        self.completion_detected_by = None
        self._reset_search_cycle(reason="level transition")

        self.telegram.notify_new_level(self.total_levels_completed, time_spent)
        self.historical_learner.record_completion(
            time_spent,
            completion_source,
        )

        logger.info(f"Level {self.total_levels_completed} completed. Time spent: {time_spent:.1f}s")
        logger.info("Waiting for unlock button after level transition")
        return State.WAIT_FOR_UNLOCK

    def _reset_search_cycle(self, reason="state reset"):
        """Reset oscillating-search progression so the next search starts from base sweep."""
        logger.debug(
            "Resetting search cycle (%s): scroll_offset_units=%.2f",
            reason,
            self.scroll_offset_units,
        )
        self.scroll_offset_units = 0
    
    def handle_wait_for_unlock(self, current_state):
        self._click_idle()
        
        # 1. PRE-ACTION DELAY: Wait on first attempt to allow animation to start/finish
        if self.wait_for_unlock_attempts == 0:
            pre_delay = getattr(config, "PRE_UNLOCK_DELAY", 0.5)
            if pre_delay > 0:
                logger.debug(f"Pre-unlock delay: waiting {pre_delay}s for UI stability")
                if self._sleep_with_interrupt(pre_delay):
                    return State.TRANSITION_LEVEL

        if config.IDLE_CLICK_SETTLE_DELAY > 0:
            if self._sleep_with_interrupt(config.IDLE_CLICK_SETTLE_DELAY):
                return State.TRANSITION_LEVEL
        
        self.wait_for_unlock_attempts += 1
        logger.debug(f"Waiting for unlock button (attempt {self.wait_for_unlock_attempts}/{self.max_wait_for_unlock_attempts})")
        
        if self.wait_for_unlock_attempts > self.max_wait_for_unlock_attempts:
            logger.warning(f"Unlock button not found after {self.max_wait_for_unlock_attempts} attempts, resetting to scroll")
            self.wait_for_unlock_attempts = 0
            return State.FIND_RED_ICONS
        
        screenshot = self._capture(max_y=config.MAX_SEARCH_Y)

        if "unlock" in self.templates:
            template, mask = self.templates["unlock"]
            found, confidence, x, y = self.image_matcher.find_template(
                screenshot, template, mask=mask,
                threshold=config.UNLOCK_THRESHOLD, template_name="unlock"
            )

            if found:
                logger.info(f"Unlock button found at ({x}, {y}) [conf: {confidence:.2f}] on attempt {self.wait_for_unlock_attempts}")
                self.mouse_controller.click(x, y, relative=True)
                if config.UNLOCK_POST_CLICK_DELAY > 0:
                    if self._sleep_with_interrupt(config.UNLOCK_POST_CLICK_DELAY):
                        return State.TRANSITION_LEVEL
                logger.info("Starting new level")
                self.wait_for_unlock_attempts = 0
                return State.FIND_RED_ICONS
        
        # 2. SMART WAITING (Variable Backoff):
        # If we fail the initial checks, wait longer to let animation finish.
        retry_delay = config.WAIT_UNLOCK_RETRY_DELAY
        backoff_threshold = getattr(config, "UNLOCK_BACKOFF_THRESHOLD", 3)
        if self.wait_for_unlock_attempts > backoff_threshold:
            retry_delay = getattr(config, "UNLOCK_MAX_RETRY_DELAY", 1.0)
            logger.debug(f"Unlock backoff active: increasing retry delay to {retry_delay}s")

        if retry_delay > 0:
            if self._sleep_with_interrupt(retry_delay):
                return State.TRANSITION_LEVEL
        return State.WAIT_FOR_UNLOCK
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        logger.info("Bot started")
        
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
        
        if config.ShowForbiddenArea and not self.overlay:
            from window_capture import ForbiddenAreaOverlay
            self.overlay = ForbiddenAreaOverlay(self.window_capture.hwnd, self.forbidden_zones)
            self.overlay.start()
            logger.info("Forbidden area overlay enabled and started")

    def stop(self):
        if not self.running:
            return
            
        self.running = False
        self._new_level_monitor_stop.set()
        if self._new_level_monitor_thread and self._new_level_monitor_thread.is_alive():
            self._new_level_monitor_thread.join(timeout=1.0)
        self.historical_learner.stop()
        if self.overlay:
            self.overlay.stop()
            self.overlay = None
        logger.info("Bot stopped")

    def step(self):
        self._clear_capture_cache()
        self._apply_tuning()
        self._enforce_state_min_interval()
        self.state_machine.update()

    def run(self):
        self.start()
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
