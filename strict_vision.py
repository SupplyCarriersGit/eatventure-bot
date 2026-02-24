import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class AssetProfile:
    """In-memory profile used by the strict 3-stage cascade verifier."""

    name: str
    template: np.ndarray
    contour: np.ndarray
    reference_area: float


def _asset_group(asset_name: str) -> str:
    return "box" if asset_name.lower().startswith("box") else "upgrade_station"


def _get_bgr_bounds_for_asset(asset_name: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    group = _asset_group(asset_name)
    if group == "box":
        return config.BOX_BROWN_BGR_LOWER, config.BOX_BROWN_BGR_UPPER
    return config.UPGRADE_STATION_BLUE_BGR_LOWER, config.UPGRADE_STATION_BLUE_BGR_UPPER


def _extract_reference_contour(template: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """
    Build a shape profile from the template itself.
    Stage-2 in production compares candidate contours against this profile via cv2.matchShapes.
    """
    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(template, lower_arr, upper_arr)

    # If strict color gating is too restrictive on the reference image, gracefully fall back
    # to a luminance-derived binary mask so we still get a geometry profile.
    if cv2.countNonZero(mask) == 0:
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def load_asset_profiles(templates: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]) -> Dict[str, AssetProfile]:
    """
    Dynamic Profile Loading (Phase 2).
    Runs once during bot initialization.

    Loads templates already scanned from Assets/ and extracts a canonical contour profile
    per asset for Stage-2 shape matching.
    """
    profiles: Dict[str, AssetProfile] = {}
    required = ["upgradeStation", "box1", "box2", "box3", "box4", "box5"]

    for asset_name in required:
        template_pack = templates.get(asset_name)
        if template_pack is None:
            logger.warning("Strict vision profile skipped (missing template): %s", asset_name)
            continue

        template, _ = template_pack
        lower, upper = _get_bgr_bounds_for_asset(asset_name)
        contour = _extract_reference_contour(template, lower, upper)

        if contour is None:
            logger.warning("Strict vision profile skipped (no contour): %s", asset_name)
            continue

        profiles[asset_name] = AssetProfile(
            name=asset_name,
            template=template,
            contour=contour,
            reference_area=float(cv2.contourArea(contour)),
        )

    logger.info("Strict vision profiles loaded: %s", ", ".join(sorted(profiles.keys())))
    return profiles


def verify_asset_strict(
    screenshot: np.ndarray,
    profile: AssetProfile,
    search_roi: Optional[Tuple[int, int, int, int]] = None,
    template_threshold: Optional[float] = None,
) -> Optional[dict]:
    """
    Strict Three-Stage Cascade Verification Pipeline:
      1) inRange color gate
      2) contour area + geometry match (cv2.matchShapes)
      3) template match on Stage-2 ROI

    Returns None unless all stages pass. On success returns center coordinates and diagnostics.
    """
    if screenshot is None or screenshot.size == 0:
        return None

    img_h, img_w = screenshot.shape[:2]
    if search_roi is None:
        x1, x2, y1, y2 = 0, img_w, 0, img_h
    else:
        x1, x2, y1, y2 = search_roi
        x1 = max(0, min(img_w, int(x1)))
        x2 = max(0, min(img_w, int(x2)))
        y1 = max(0, min(img_h, int(y1)))
        y2 = max(0, min(img_h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = screenshot[y1:y2, x1:x2]
    lower, upper = _get_bgr_bounds_for_asset(profile.name)

    # ---- Stage 1: Pixel/Color Matching (Broad Net) ----
    binary_mask = cv2.inRange(roi, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    if cv2.countNonZero(binary_mask) == 0:
        return None

    # ---- Stage 2: Contour & Shape Matching (Geometry Filter) ----
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_area = float(roi.shape[0] * roi.shape[1])
    min_area = float(getattr(config, "STRICT_ASSET_MIN_CONTOUR_AREA", 180))
    max_area_ratio = float(getattr(config, "STRICT_ASSET_MAX_CONTOUR_AREA_RATIO", 0.35))
    max_area = roi_area * max_area_ratio
    shape_tolerance = float(getattr(config, "STRICT_ASSET_SHAPE_TOLERANCE", 0.16))

    candidate_boxes = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        score = cv2.matchShapes(profile.contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if score > shape_tolerance:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        candidate_boxes.append((score, bx, by, bw, bh))

    if not candidate_boxes:
        return None

    # ---- Stage 3: Image/Template Matching (Final Authenticator) ----
    threshold = (
        float(template_threshold)
        if template_threshold is not None
        else float(getattr(config, "STRICT_ASSET_TEMPLATE_THRESHOLD", 0.88))
    )

    best_payload = None
    template = profile.template
    th, tw = template.shape[:2]

    for shape_score, bx, by, bw, bh in sorted(candidate_boxes, key=lambda item: item[0]):
        # Stage-3 requirement: use the exact Stage-2 contour ROI, converted to a square crop
        # around the detected geometry center. This preserves local context and avoids full-frame
        # template matching that can drift into environmental false positives.
        cx = bx + (bw // 2)
        cy = by + (bh // 2)
        side = max(bw, bh)
        half = side // 2

        sx1 = max(0, cx - half)
        sy1 = max(0, cy - half)
        sx2 = min(roi.shape[1], sx1 + side)
        sy2 = min(roi.shape[0], sy1 + side)
        square_crop = roi[sy1:sy2, sx1:sx2]
        if square_crop.size == 0:
            continue

        # matchTemplate requires the search image to be >= template size.
        # If needed, pad the candidate square (edge-replicated) so we can still perform
        # template verification without changing Stage-2 localization.
        ch, cw = square_crop.shape[:2]
        if ch < th or cw < tw:
            pad_y = max(0, th - ch)
            pad_x = max(0, tw - cw)
            top = pad_y // 2
            bottom = pad_y - top
            left = pad_x // 2
            right = pad_x - left
            square_crop = cv2.copyMakeBorder(square_crop, top, bottom, left, right, cv2.BORDER_REPLICATE)

        tm_result = cv2.matchTemplate(square_crop, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, max_loc = cv2.minMaxLoc(tm_result)

        if confidence < threshold:
            continue

        abs_x1 = x1 + sx1 + int(max_loc[0])
        abs_y1 = y1 + sy1 + int(max_loc[1])
        center_x = abs_x1 + (tw // 2)
        center_y = abs_y1 + (th // 2)

        payload = {
            "asset_name": profile.name,
            "center_x": int(center_x),
            "center_y": int(center_y),
            "bbox": (int(abs_x1), int(abs_y1), int(tw), int(th)),
            "template_confidence": float(confidence),
            "shape_score": float(shape_score),
        }

        if best_payload is None or payload["template_confidence"] > best_payload["template_confidence"]:
            best_payload = payload

    return best_payload
