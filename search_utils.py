import time
import logging
import config
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

class OscillatingSearcher:
    """
    Refactored Algorithm Engine: Implements the Interleaved Arithmetic Progression 
    Search Strategy with high-priority interrupt sensitivity.
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self.max_cycles = getattr(config, "MAX_SCROLL_CYCLES", 15)
        self.scroll_increment = getattr(config, "SCROLL_INCREMENT_STEP", 2)
        self.settle_duration = getattr(config, "OSCILLATION_SETTLE_TIME", 0.5)

    def execute_cycle(self, 
                      check_priority: Callable, 
                      check_main_target: Callable, 
                      check_fallbacks: Optional[Callable] = None) -> Optional[Any]:
        """
        The orchestrator for the widening search pattern.
        Requirement: maintain exact logic flow (Baseline Scan -> Outer Cycles -> Interleaved Inner Loop).
        """
        logger.info(f"[Search] Initializing Interleaved Search (Limit: {self.max_cycles} cycles)")

        # Baseline: Verify current area before moving
        initial_hit = self._perform_vision_pass(check_priority, check_main_target, check_fallbacks)
        if initial_hit:
            return initial_hit

        search_direction = 1 # 1: Down (Drag UP), -1: Up (Drag DOWN)
        for cycle_index in range(self.max_cycles):
            # Arithmetic Progression Formula: 1, 3, 5, 7...
            steps_in_cycle = 1 + (cycle_index * self.scroll_increment)
            
            logger.info(f"[Search] Cycle {cycle_index + 1}: Executing {steps_in_cycle} steps")

            target_found = self._run_step_sequence(steps_in_cycle, search_direction, 
                                                  check_priority, check_main_target, check_fallbacks)
            if target_found:
                return target_found

            # Post-Sequence stabilization before reversing direction
            self.bot.sleep(getattr(config, "CYCLE_PAUSE_DURATION", 0.5))
            search_direction *= -1

        logger.warning(f"[Search] Logic exhausted after {self.max_cycles} cycles.")
        return None

    def _run_step_sequence(self, count: int, direction: int, p_check: Callable, m_check: Callable, f_check: Optional[Callable]) -> Optional[Any]:
        """Executes a sequence of individual scroll-and-scan steps."""
        for _ in range(count):
            # Mechanical Guard: Check bot status before interaction
            if not self.bot.running:
                return None

            # 1. Action
            self.perform_scroll(direction)
            
            # 2. Wait (Stabilization)
            self.bot.sleep(getattr(config, "SCROLL_INTERVAL_PAUSE", 0.4))

            # 3. Scan
            hit = self._perform_vision_pass(p_check, m_check, f_check)
            if hit:
                return hit
        return None

    def _perform_vision_pass(self, p_check: Callable, m_check: Callable, f_check: Optional[Callable]) -> Optional[Any]:
        """Atomic vision check following the Priority -> Main -> Fallback protocol."""
        priority_hit = p_check()
        if priority_hit:
            return priority_hit

        main_hit = m_check()
        if main_hit:
            return main_hit

        if f_check:
            f_check()
            
        return None

    def perform_scroll(self, direction: Any, distance_ratio: Optional[float] = None, duration: float = 0.5):
        """Standardized mechanical drag interface."""
        dir_int = self._map_direction(direction)
        start_x, start_y = getattr(config, "SCROLL_START_POS", (812, 540))
        
        # Distance calculation
        ratio = distance_ratio or getattr(config, "SCROLL_DISTANCE_RATIO", 1.0)
        pixel_distance = int(getattr(config, "SCROLL_PIXEL_STEP", 150) * ratio)
        end_y = start_y - (pixel_distance * dir_int)

        self.bot.mouse_controller.drag(
            start_x, start_y, start_x, end_y, 
            duration=duration, relative=True,
            interrupt_check=lambda: self.bot.check_critical_interrupts(raise_exception=False)
        )
        
        if hasattr(self.bot, 'scroll_offset_units'):
            self.bot.scroll_offset_units -= (ratio * dir_int)

    def _map_direction(self, direction: Any) -> int:
        """Standardizes input directions into integers (1 or -1)."""
        if isinstance(direction, int): 
            return direction
        return {"DOWN": 1, "UP": -1}.get(str(direction).upper(), 1)
