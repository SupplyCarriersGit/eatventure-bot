# Phase 1 Diagnostic Report: Strict Cascade Integration Plan

## Holistic audit (vision utilities + handlers + FSM routing)

### Vision utilities before refactor
1. `image_matcher.find_template()` was the primary single-stage detector for boxes/stations.
2. `handlers.BaseHandler.verify_bgr_match()` was a local whitelist check, but still template-only and not a true 3-stage geometric funnel.
3. `bot.py` fallback and state handlers also used direct template matching for box/station actions.

### Handler layer before refactor
- `UpgradeStationHandler.process()` consumed candidate points from `self.bot._find_upgrade_stations(...)`, then called `verify_bgr_match(...)`.
- `BoxHandler.process()` consumed `self.bot._find_boxes(...)`, then looped `verify_bgr_match` over `box1..box5`.
- Net effect: candidate generation + verification both remained template-centric and vulnerable to environment lookalikes.

### FSM routing and click gatekeeper audit
- Hub-and-spoke FSM transitions are controlled in `bot.py` state handlers and `state_machine.py`.
- Coordinate safety is enforced by mouse click resolution via `mouse_controller.is_safe_to_click(...)` / forbidden zones.
- Required invariants to preserve:
  - Do not alter state transition graph.
  - Do not alter hold/paging mechanics.
  - Do not bypass safe-click gatekeeper.

## Exact rip-out and injection map

### Rip-out targets (single-stage logic)
1. `handlers.py`:
   - `UpgradeStationHandler.process()` old `_find_upgrade_stations + verify_bgr_match` flow.
   - `BoxHandler.process()` old `_find_boxes + verify_bgr_match` flow.
2. `bot.py`:
   - `_scan_and_click_non_red_assets()` old direct template matching.
   - `handle_search_upgrade_station()` old direct template matching.
   - `handle_open_boxes()` old direct template matching.

### Injection points (strict funnel)
1. Init-time profile load (once):
   - `self.asset_profiles = load_asset_profiles(self.templates)` in bot initialization.
2. Runtime strict verifier:
   - `verify_asset_strict(...)` now drives all box/station clicks in the above paths.
3. Handler integration requirement:
   - `UpgradeStationHandler` and `BoxHandler` now exclusively use strict verification.

## Three-stage cascade design contract
1. **Stage 1: Pixel/Color Gate**
   - `cv2.inRange` with strict BGR bounds (asset class specific).
   - Empty mask => hard fail.
2. **Stage 2: Contour/Shape Gate**
   - `cv2.findContours` on Stage-1 mask.
   - Bounding-box area filters remove floor/water mega-contours and micro-noise.
   - `cv2.matchShapes` compares candidates with loaded profile contour.
   - No geometric match => hard fail.
3. **Stage 3: Template Gate**
   - Crop the Stage-2-passed local ROI as a square candidate window.
   - Run `cv2.matchTemplate` against loaded asset image.
   - Confidence below threshold => hard fail.
   - Otherwise return monitor-click payload center.

## Reliability notes
- The funnel is intentionally conservative and sequential.
- Click authorization requires all three gates in order.
- FSM routing, scroll counters, hold mechanics, and safe-click gatekeeper remain untouched.
