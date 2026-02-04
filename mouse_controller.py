import win32api
import win32con
import win32gui
import time
import logging
import config

logger = logging.getLogger(__name__)


class MouseController:
    def __init__(self, hwnd, click_delay=0.1):
        self.hwnd = hwnd
        self.click_delay = click_delay
        self._last_click_time = 0.0
        self._last_cursor_pos = None
        self._last_drag_time = 0.0

    def _resolve_screen_position(self, x, y, relative=True, check_forbidden=True):
        if relative:
            if check_forbidden and self.is_in_forbidden_zone(x, y):
                return None

            win_x, win_y = self.get_window_position()
            screen_x = win_x + x
            screen_y = win_y + y
        else:
            screen_x = x
            screen_y = y

        return int(screen_x), int(screen_y)

    def _send_click(self, screen_x, screen_y, down_up_delay=None):
        if self._should_move_cursor(screen_x, screen_y):
            self._move_cursor(screen_x, screen_y)

        self._ensure_cursor_at_target(screen_x, screen_y)
        self._correct_cursor_position(screen_x, screen_y)
        self._ensure_min_click_interval()

        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
        time.sleep(config.MOUSE_DOWN_UP_DELAY if down_up_delay is None else down_up_delay)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
        self._last_click_time = time.monotonic()

    def _send_mouse_down(self, screen_x, screen_y):
        if self._should_move_cursor(screen_x, screen_y):
            self._move_cursor(screen_x, screen_y)

        self._ensure_cursor_at_target(screen_x, screen_y)
        self._correct_cursor_position(screen_x, screen_y)
        self._ensure_min_click_interval()
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)

    def _send_mouse_up(self, screen_x, screen_y):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
        self._last_click_time = time.monotonic()

    def _ensure_min_click_interval(self):
        min_interval = getattr(config, "MIN_CLICK_INTERVAL", 0.0)
        if min_interval <= 0:
            return
        now = time.monotonic()
        wait_time = self._last_click_time + min_interval - now
        if wait_time > 0:
            time.sleep(wait_time)

    def _ensure_min_drag_interval(self):
        min_interval = getattr(config, "SCROLL_MIN_INTERVAL", 0.0)
        if min_interval <= 0:
            return
        now = time.monotonic()
        wait_time = self._last_drag_time + min_interval - now
        if wait_time > 0:
            time.sleep(wait_time)
        self._last_drag_time = time.monotonic()

    def _correct_cursor_position(self, screen_x, screen_y):
        retries = max(0, getattr(config, "MOUSE_TARGET_RETRIES", 0))
        if retries <= 0:
            return
        tolerance = getattr(config, "MOUSE_POSITION_TOLERANCE", 0)
        correction_delay = getattr(config, "MOUSE_TARGET_CORRECTION_DELAY", 0.0)
        target = (int(screen_x), int(screen_y))

        for _ in range(retries):
            current = win32api.GetCursorPos()
            if abs(current[0] - target[0]) <= tolerance and abs(current[1] - target[1]) <= tolerance:
                self._last_cursor_pos = target
                return
            win32api.SetCursorPos(target)
            if correction_delay > 0:
                time.sleep(correction_delay)
        self._last_cursor_pos = target

    def _should_move_cursor(self, screen_x, screen_y):
        if self._last_cursor_pos is None:
            return True
        tolerance = getattr(config, "MOUSE_POSITION_TOLERANCE", 0)
        dx = abs(self._last_cursor_pos[0] - screen_x)
        dy = abs(self._last_cursor_pos[1] - screen_y)
        return dx > tolerance or dy > tolerance

    def _move_cursor(self, screen_x, screen_y):
        target = (int(screen_x), int(screen_y))
        retries = max(1, getattr(config, "MOUSE_MOVE_RETRIES", 1))
        retry_delay = getattr(config, "MOUSE_MOVE_RETRY_DELAY", 0.0)
        tolerance = getattr(config, "MOUSE_POSITION_TOLERANCE", 0)

        for _ in range(retries):
            win32api.SetCursorPos(target)
            if retry_delay > 0:
                time.sleep(retry_delay)
            current = win32api.GetCursorPos()
            if abs(current[0] - target[0]) <= tolerance and abs(current[1] - target[1]) <= tolerance:
                break

        time.sleep(config.MOUSE_MOVE_DELAY)
        self._last_cursor_pos = target

    def _ensure_cursor_at_target(self, screen_x, screen_y):
        target = (int(screen_x), int(screen_y))
        tolerance = getattr(config, "MOUSE_POSITION_TOLERANCE", 0)
        timeout = getattr(config, "MOUSE_TARGET_TIMEOUT", 0.0)
        check_interval = getattr(config, "MOUSE_TARGET_CHECK_INTERVAL", 0.0)
        settle_delay = getattr(config, "MOUSE_TARGET_SETTLE_DELAY", 0.0)
        hover_delay = getattr(config, "MOUSE_TARGET_HOVER_DELAY", 0.0)
        stabilize_duration = getattr(config, "MOUSE_STABILIZE_DURATION", 0.0)

        start_time = time.monotonic()
        stable_since = None
        while True:
            current = win32api.GetCursorPos()
            if abs(current[0] - target[0]) <= tolerance and abs(current[1] - target[1]) <= tolerance:
                if stable_since is None:
                    stable_since = time.monotonic()
                if stabilize_duration <= 0 or time.monotonic() - stable_since >= stabilize_duration:
                    if settle_delay > 0:
                        time.sleep(settle_delay)
                    if hover_delay > 0:
                        time.sleep(hover_delay)
                    self._last_cursor_pos = target
                    return
            else:
                stable_since = None

            if timeout <= 0 or time.monotonic() - start_time >= timeout:
                win32api.SetCursorPos(target)
                self._last_cursor_pos = target
                if hover_delay > 0:
                    time.sleep(hover_delay)
                return

            if check_interval > 0:
                time.sleep(check_interval)
    
    def is_in_forbidden_zone(self, x, y):
        if (y >= config.FORBIDDEN_CLICK_Y_MIN and 
            config.FORBIDDEN_CLICK_X_MIN <= x <= config.FORBIDDEN_CLICK_X_MAX):
            logger.warning(f"Coordinates ({x}, {y}) blocked - FORBIDDEN_CLICK zone")
            return True
        
        if (config.FORBIDDEN_ZONE_1_Y_MIN <= y <= config.FORBIDDEN_ZONE_1_Y_MAX and
            config.FORBIDDEN_ZONE_1_X_MIN <= x <= config.FORBIDDEN_ZONE_1_X_MAX):
            logger.warning(f"Coordinates ({x}, {y}) blocked - FORBIDDEN_ZONE_1")
            return True
        
        if (config.FORBIDDEN_ZONE_2_Y_MIN <= y <= config.FORBIDDEN_ZONE_2_Y_MAX and
            config.FORBIDDEN_ZONE_2_X_MIN <= x <= config.FORBIDDEN_ZONE_2_X_MAX):
            logger.warning(f"Coordinates ({x}, {y}) blocked - FORBIDDEN_ZONE_2")
            return True
        
        if (config.FORBIDDEN_ZONE_3_Y_MIN <= y <= config.FORBIDDEN_ZONE_3_Y_MAX and
            config.FORBIDDEN_ZONE_3_X_MIN <= x <= config.FORBIDDEN_ZONE_3_X_MAX):
            logger.warning(f"Coordinates ({x}, {y}) blocked - FORBIDDEN_ZONE_3")
            return True
        
        if (config.FORBIDDEN_ZONE_4_Y_MIN <= y <= config.FORBIDDEN_ZONE_4_Y_MAX and
            config.FORBIDDEN_ZONE_4_X_MIN <= x <= config.FORBIDDEN_ZONE_4_X_MAX):
            logger.warning(f"Coordinates ({x}, {y}) blocked - FORBIDDEN_ZONE_4")
            return True
        
        if (config.FORBIDDEN_ZONE_5_Y_MIN <= y <= config.FORBIDDEN_ZONE_5_Y_MAX and
            config.FORBIDDEN_ZONE_5_X_MIN <= x <= config.FORBIDDEN_ZONE_5_X_MAX):
            logger.warning(f"Coordinates ({x}, {y}) blocked - FORBIDDEN_ZONE_5")
            return True

        if (config.FORBIDDEN_ZONE_6_Y_MIN <= y <= config.FORBIDDEN_ZONE_6_Y_MAX and
            config.FORBIDDEN_ZONE_6_X_MIN <= x <= config.FORBIDDEN_ZONE_6_X_MAX):
            logger.warning(f"Coordinates ({x}, {y}) blocked - FORBIDDEN_ZONE_6")
            return True
        
        return False
    
    def get_window_position(self):
        x, y = win32gui.ClientToScreen(self.hwnd, (0, 0))
        return x, y
    
    def move_to(self, x, y, relative=True):
        if relative:
            win_x, win_y = self.get_window_position()
            screen_x = win_x + x
            screen_y = win_y + y
        else:
            screen_x = x
            screen_y = y
        
        win32api.SetCursorPos((int(screen_x), int(screen_y)))
        self._last_cursor_pos = (int(screen_x), int(screen_y))
        logger.info(f"Cursor moved to window position ({x}, {y})")
    
    def click(self, x, y, relative=True, delay=None, wait_after=True):
        screen_pos = self._resolve_screen_position(x, y, relative=relative)
        if screen_pos is None:
            if wait_after:
                time.sleep(self.click_delay if delay is None else delay)
            return False

        screen_x, screen_y = screen_pos
        self._send_click(screen_x, screen_y)

        logger.info(f"Clicked at ({screen_x}, {screen_y})")

        if wait_after:
            time.sleep(self.click_delay if delay is None else delay)
        return True

    def mouse_down(self, x, y, relative=True):
        screen_pos = self._resolve_screen_position(x, y, relative=relative)
        if screen_pos is None:
            return False

        screen_x, screen_y = screen_pos
        self._send_mouse_down(screen_x, screen_y)
        self._last_cursor_pos = (screen_x, screen_y)
        logger.info(f"Mouse down at ({screen_x}, {screen_y})")
        return True

    def mouse_up(self, x, y, relative=True):
        screen_pos = self._resolve_screen_position(x, y, relative=relative, check_forbidden=False)
        if screen_pos is None:
            return False

        screen_x, screen_y = screen_pos
        self._send_mouse_up(screen_x, screen_y)
        self._last_cursor_pos = (screen_x, screen_y)
        logger.info(f"Mouse up at ({screen_x}, {screen_y})")
        return True
    
    def double_click(self, x, y, relative=True):
        self.click(x, y, relative)
        time.sleep(config.DOUBLE_CLICK_DELAY)
        self.click(x, y, relative)
    
    def hold_at(self, x, y, duration=None, relative=True):
        if duration is None:
            duration = config.UPGRADE_HOLD_DURATION

        screen_pos = self._resolve_screen_position(x, y, relative=relative)
        if screen_pos is None:
            return False

        screen_x, screen_y = screen_pos

        logger.info(
            "Holding click at (%s, %s) for %ss",
            screen_x,
            screen_y,
            duration,
        )
        self._send_mouse_down(screen_x, screen_y)
        time.sleep(duration)
        self._send_mouse_up(screen_x, screen_y)
        time.sleep(self.click_delay)
        return True
    
    def drag(self, from_x, from_y, to_x, to_y, duration=0.3, relative=True):
        if relative:
            win_x, win_y = self.get_window_position()
            screen_from_x = win_x + from_x
            screen_from_y = win_y + from_y
            screen_to_x = win_x + to_x
            screen_to_y = win_y + to_y
        else:
            screen_from_x = from_x
            screen_from_y = from_y
            screen_to_x = to_x
            screen_to_y = to_y

        self._ensure_min_drag_interval()
        
        win32api.SetCursorPos((int(screen_from_x), int(screen_from_y)))
        self._ensure_cursor_at_target(int(screen_from_x), int(screen_from_y))
        self._correct_cursor_position(int(screen_from_x), int(screen_from_y))
        self._last_cursor_pos = (int(screen_from_x), int(screen_from_y))
        time.sleep(config.MOUSE_MOVE_DELAY)
        
        win32api.mouse_event(
            win32con.MOUSEEVENTF_LEFTDOWN,
            int(screen_from_x),
            int(screen_from_y),
            0,
            0,
        )
        time.sleep(config.MOUSE_DOWN_UP_DELAY)
        
        steps = max(1, int(getattr(config, "SCROLL_STEP_COUNT", 20)))
        duration = max(duration, 0.001)
        start_time = time.monotonic()
        for i in range(steps + 1):
            t = i / steps
            current_x = int(screen_from_x + (screen_to_x - screen_from_x) * t)
            current_y = int(screen_from_y + (screen_to_y - screen_from_y) * t)
            win32api.SetCursorPos((current_x, current_y))
            target_time = start_time + (duration * t)
            sleep_time = target_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        win32api.mouse_event(
            win32con.MOUSEEVENTF_LEFTUP,
            int(screen_to_x),
            int(screen_to_y),
            0,
            0,
        )
        self._ensure_cursor_at_target(int(screen_to_x), int(screen_to_y))
        self._correct_cursor_position(int(screen_to_x), int(screen_to_y))
        self._last_cursor_pos = (int(screen_to_x), int(screen_to_y))
        logger.info(f"Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y})")
        settle_delay = getattr(config, "SCROLL_SETTLE_DELAY", 0.0)
        time.sleep(settle_delay if settle_delay > 0 else self.click_delay)
