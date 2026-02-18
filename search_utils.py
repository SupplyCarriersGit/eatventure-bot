import time
import logging
import config
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

class OscillatingSearcher:
    """
    Principal Architect Implementation: OscillatingSearcher
    A robust, fail-safe searching engine designed to systematically explore 
    a viewport while preventing logic loops and vision blindness.
    """

    def __init__(self, bot: Any):
        """
        Initializes the searcher with canonical configurations.
        :param bot: The main bot instance providing vision and mouse control.
        """
        self.bot = bot
        self.max_retries = getattr(config, "OSCILLATION_MAX_RETRIES", 10)
        self.start_pos = getattr(config, "SCROLL_START_POS", (812, 540))
        self.base_step = getattr(config, "SCROLL_PIXEL_STEP", 150)
        self.ratio = getattr(config, "SCROLL_DISTANCE_RATIO", 1.0)
        self.settle_time = getattr(config, "OSCILLATION_SETTLE_TIME", 0.5)

    def perform_scroll(self, direction: Any, distance_ratio: Optional[float] = None, duration: float = 0.5):
        """
        Requirement: The "Click vs. Drag" Fix.
        Executes a functional Drag Gesture (Mouse Down -> Move -> Mouse Up).
        :param direction: 1 or 'DOWN' for Scroll Down, -1 or 'UP' for Scroll Up.
        :param distance_ratio: Optional multiplier to override default scroll distance.
        :param duration: Time in seconds for the swipe movement.
        """
        # Map string directions to integers
        if isinstance(direction, str):
            dir_map = {"DOWN": 1, "UP": -1}
            dir_int = dir_map.get(direction.upper(), 1)
        else:
            dir_int = direction

        start_x, start_y = self.start_pos
        ratio = distance_ratio if distance_ratio is not None else self.ratio
        dist = int(self.base_step * ratio)
        
        # Directional Logic:
        # Scroll Down (1) -> Drag UP (y decreases)
        # Scroll Up (-1) -> Drag DOWN (y increases)
        end_y = start_y - (dist * dir_int)

        logger.info(f"[Scroll] Executing Drag: ({start_x}, {start_y}) -> ({start_x}, {end_y}) [dir={direction}, ratio={ratio:.2f}]")
        
        # Use the built-in drag method which handles down, steps, duration, and up.
        success = self.bot.mouse_controller.drag(
            start_x, start_y, 
            start_x, end_y, 
            duration=duration, 
            relative=True
        )
        
        if success:
            # Update bot's internal drift tracking for drift correction
            if hasattr(self.bot, 'scroll_offset_units'):
                self.bot.scroll_offset_units -= (ratio * dir_int)

    def execute_cycle(self, 
                      check_priority: Callable, 
                      check_main_target: Callable, 
                      check_fallbacks: Optional[Callable] = None) -> Optional[Any]:
        """
        Senior Algorithm Implementation: Arithmetic Progression Search Strategy.
        Widens the search area each cycle (1, 3, 5, 7...) to prevent local traps.
        """
        max_cycles = getattr(config, "MAX_SCROLL_CYCLES", 15)
        increment = getattr(config, "SCROLL_INCREMENT_STEP", 2)
        direction = 1  # 1 for Scroll Down (Drag UP), -1 for Scroll Up (Drag DOWN)
        
        logger.info(f"[Search] Starting Arithmetic Progression Search (Max Cycles: {max_cycles})")

        for current_cycle in range(max_cycles):
            # Calculate arithmetic progression: 1, 3, 5, 7, 9...
            scrolls_to_perform = 1 + (current_cycle * increment)
            logger.info(f"[Search] Cycle {current_cycle + 1}/{max_cycles}: performing {scrolls_to_perform} steps (Dir: {direction})")

            # --- INNER LOOP: Sequential Step-Scan ---
            for step in range(scrolls_to_perform):
                # INTERRUPT CHECK: Ensure immediate stop
                if hasattr(self.bot, 'running') and not self.bot.running:
                    logger.info("[Search] Interrupt detected; aborting search cycle")
                    return None

                # STEP A & B: Priority & Main Target Scan (The "Scan-First" Requirement)
                priority_result = check_priority()
                if priority_result:
                    logger.info(f"[Search] Priority target found during cycle {current_cycle + 1}, step {step + 1}")
                    return priority_result

                main_result = check_main_target()
                if main_result:
                    logger.info(f"[Search] Main target acquired!")
                    return main_result

                # STEP C: Fallback Scan (Side Effects)
                if check_fallbacks:
                    check_fallbacks()

                # STEP D: Mechanical Scroll
                self.perform_scroll(direction)
                
                # STEP E: Settle Wait (Interval Pause)
                # Requirement: Inside inner loop, strictly after scroll.
                time.sleep(getattr(config, "SCROLL_INTERVAL_PAUSE", 0.5))

            # --- CYCLE COMPLETION ---
            # Requirement: CYCLE_PAUSE_DURATION after the sequence finishes.
            logger.debug(f"[Search] Cycle {current_cycle + 1} sequence complete. Stabilizing...")
            time.sleep(getattr(config, "CYCLE_PAUSE_DURATION", 0.5))
            
            # Widen the area by flipping direction for the next (larger) cycle
            direction *= -1

        logger.warning(f"[Search] Search strategy exhausted after {max_cycles} cycles.")
        return None
