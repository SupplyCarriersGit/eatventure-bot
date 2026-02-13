import json
import os
import time
import logging
import threading
import re
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
        self.box_threshold = config.BOX_THRESHOLD
        self.persistence = persistence
        self._lock = threading.Lock()
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
        boost = max(0.0, min(1.0, (confidence - 0.8))) * self.confidence_boost
        return min(self.alpha + boost, self.alpha_max)

    def update_red_icon_confidences(self, confidences):
        if not self.enabled or not confidences:
            return
        with self._lock:
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
        with self._lock:
            if confidences:
                self._red_icon_miss_count = 0
                avg_conf = sum(confidences) / len(confidences)
                target = max(
                    config.AI_RED_ICON_THRESHOLD_MIN,
                    min(avg_conf - config.AI_RED_ICON_MARGIN, config.AI_RED_ICON_THRESHOLD_MAX),
                )
                self.red_icon_threshold = self._ema(self.red_icon_threshold, target, self._adaptive_alpha(avg_conf))
                self._persist()
                return

            self._red_icon_miss_count += 1
            if self._red_icon_miss_count < config.AI_RED_ICON_MISS_WINDOW:
                return
            self._red_icon_miss_count = 0

            current_min = getattr(config, "AI_RED_ICON_THRESHOLD_MIN", 0.85)
            if self.red_icon_threshold <= current_min:
                return

            target = max(
                current_min,
                self.red_icon_threshold - config.AI_RED_ICON_MISS_STEP,
            )
            self.red_icon_threshold = self._ema(self.red_icon_threshold, target, self.alpha_max)
            self._persist()

    def update_new_level_confidence(self, confidence):
        if not self.enabled or confidence <= 0:
            return
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._box_miss_count = 0
            min_thresh = getattr(config, "AI_RED_ICON_THRESHOLD_MIN", 0.85)
            target = max(
                min_thresh,
                min(confidence, 0.995),
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._box_miss_count += 1
            if self._box_miss_count < 3:
                return
            self._box_miss_count = 0
            min_thresh = getattr(config, "AI_RED_ICON_THRESHOLD_MIN", 0.85)
            target = max(min_thresh, self.box_threshold - 0.005)
            self.box_threshold = self._ema(self.box_threshold, target, self.alpha_max)
            self._persist()

    def apply_persisted_state(self, state):
        if not state:
            return
        with self._lock:
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

    def _persist(self):
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
        self.persistence.save({key: float(value) for key, value in state.items()})


class VisionPersistence:
    def __init__(self, path, save_interval):
        self.path = path
        self.save_interval = save_interval
        self._last_save_time = 0.0
        self._lock = threading.Lock()

    def load(self):
        if not self.path:
            return {}
        if not os.path.exists(self.path):
            return {}
        try:
            with self._lock:
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
        with self._lock:
            if self.save_interval > 0 and now - self._last_save_time < self.save_interval:
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

            max_pair_processed = self._total_completions // self.pair_window
            max_batch_processed = self._total_completions // self.batch_window
            self._last_pair_processed = min(self._last_pair_processed, max_pair_processed)
            self._last_batch_processed = min(self._last_batch_processed, max_batch_processed)

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
            self._thread.join(timeout=1.0)
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
            self._records = self._records[-120:]
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

        self.bot.apply_learned_behavior(tuned, reason=label, best_time=record.get("time_spent", 0.0))

        if self.auto_write_config:
            try:
                self._write_config_updates(tuned)
            except OSError as exc:
                logger.warning("Historical learner could not update config.py: %s", exc)
            except Exception:
                logger.exception("Unexpected error while persisting learned config")

    def _write_config_updates(self, tuned):
        mapping = {
            "CLICK_DELAY": tuned.get("click_delay"),
            "MOUSE_MOVE_DELAY": tuned.get("move_delay"),
            "UPGRADE_CLICK_INTERVAL": tuned.get("upgrade_click_interval"),
            "UPGRADE_SEARCH_INTERVAL": tuned.get("search_interval"),
        }
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        if not os.path.exists(config_path):
            return

        with open(config_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        changed = False
        for key, value in mapping.items():
            if value is None:
                continue
            replacement = f"{key} = {float(value):.6f}"
            pattern = rf"^{key}\s*=\s*[^\n]+"
            updated, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
            if count:
                content = updated
                changed = True

        if changed:
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(content)

    def _persist(self):
        if not self.persistence:
            return
        state = {
            "records": self._records[-120:],
            "total_completions": self._total_completions,
            "last_pair_processed": self._last_pair_processed,
            "last_batch_processed": self._last_batch_processed,
        }
        self.persistence.save(state)
