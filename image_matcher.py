import cv2
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemplateData:
    name: str
    color: np.ndarray
    gray: np.ndarray
    mask: np.ndarray | None


class ImageMatcher:
    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def to_gray(self, image):
        if image is None:
            return None
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def load_template(self, template_path):
        template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if template is None:
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        mask = None
        if len(template.shape) == 3 and template.shape[2] == 4:
            alpha = template[:, :, 3]
            mask = np.zeros_like(alpha)
            mask[alpha > 0] = 255
            template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

        gray = self.to_gray(template)
        return TemplateData(
            name=template_path.stem,
            color=template,
            gray=gray,
            mask=mask,
        )
    
    def find_template(
        self,
        screenshot,
        template: TemplateData,
        mask=None,
        threshold=None,
        template_name="Unknown",
        check_color=False,
        screenshot_gray=None,
    ):
        thresh = threshold if threshold else self.threshold

        template_mask = mask if mask is not None else template.mask
        template_name = template_name or template.name
        screenshot_gray = screenshot_gray if screenshot_gray is not None else self.to_gray(screenshot)
        template_gray = template.gray

        if template_gray.shape[0] > screenshot_gray.shape[0] or template_gray.shape[1] > screenshot_gray.shape[1]:
            logger.debug(
                "Template is larger than screenshot. "
                f"Template: {template_gray.shape}, Screenshot: {screenshot_gray.shape}"
            )
            return False, 0.0, 0, 0

        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_SQDIFF_NORMED, mask=template_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        confidence = 1 - min_val
        
        if confidence >= thresh:
            h, w = template_gray.shape[:2]
            center_x = min_loc[0] + w // 2
            center_y = min_loc[1] + h // 2
            
            if check_color:
                color_match = self._check_color_similarity(
                    screenshot,
                    template,
                    min_loc,
                    template_mask,
                )
                if not color_match:
                    logger.debug(f"[{template_name}] Color check failed at ({center_x}, {center_y}), confidence: {confidence:.2%}")
                    return False, confidence, 0, 0
            
            return True, confidence, center_x, center_y
        
        return False, confidence, 0, 0
    
    def _check_color_similarity(self, screenshot, template: TemplateData, location, mask=None):
        x, y = location
        h, w = template.color.shape[:2]
        
        roi = screenshot[y:y+h, x:x+w]
        
        if roi.shape[:2] != template.color.shape[:2]:
            return True
        
        if mask is not None:
            template_masked = cv2.bitwise_and(template.color, template.color, mask=mask)
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
        else:
            template_masked = template.color
            roi_masked = roi
        
        hist_template_b = cv2.calcHist([template_masked], [0], mask, [32], [0, 256])
        hist_template_g = cv2.calcHist([template_masked], [1], mask, [32], [0, 256])
        hist_template_r = cv2.calcHist([template_masked], [2], mask, [32], [0, 256])
        
        hist_roi_b = cv2.calcHist([roi_masked], [0], mask, [32], [0, 256])
        hist_roi_g = cv2.calcHist([roi_masked], [1], mask, [32], [0, 256])
        hist_roi_r = cv2.calcHist([roi_masked], [2], mask, [32], [0, 256])
        
        cv2.normalize(hist_template_b, hist_template_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_template_g, hist_template_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_template_r, hist_template_r, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_roi_b, hist_roi_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_roi_g, hist_roi_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_roi_r, hist_roi_r, 0, 1, cv2.NORM_MINMAX)
        
        corr_b = cv2.compareHist(hist_template_b, hist_roi_b, cv2.HISTCMP_CORREL)
        corr_g = cv2.compareHist(hist_template_g, hist_roi_g, cv2.HISTCMP_CORREL)
        corr_r = cv2.compareHist(hist_template_r, hist_roi_r, cv2.HISTCMP_CORREL)
        
        avg_corr = (corr_b + corr_g + corr_r) / 3
        
        color_threshold = 0.7
        return avg_corr >= color_threshold
    
    def find_all_templates(
        self,
        screenshot,
        template: TemplateData,
        mask=None,
        threshold=None,
        min_distance=15,
        scales=None,
        template_name="Unknown",
        screenshot_gray=None,
    ):
        thresh = threshold if threshold else self.threshold
        all_matches = []
        
        if scales is None:
            scales = [1.0]

        template_mask = mask if mask is not None else template.mask
        template_name = template_name or template.name
        screenshot_gray = screenshot_gray if screenshot_gray is not None else self.to_gray(screenshot)
        template_gray = template.gray

        if template_gray.shape[0] > screenshot_gray.shape[0] or template_gray.shape[1] > screenshot_gray.shape[1]:
            logger.debug(
                "Template is larger than screenshot. "
                f"Template: {template_gray.shape}, Screenshot: {screenshot_gray.shape}"
            )
            return []
        
        for scale in scales:
            if scale != 1.0:
                scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                scaled_mask = None
                if template_mask is not None:
                    scaled_mask = cv2.resize(template_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    scaled_mask[scaled_mask > 0] = 255
            else:
                scaled_template = template_gray
                scaled_mask = template_mask
            
            if scaled_template.shape[0] > screenshot_gray.shape[0] or scaled_template.shape[1] > screenshot_gray.shape[1]:
                continue
            
            result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_SQDIFF_NORMED, mask=scaled_mask)
            
            locations = np.where(result <= (1 - thresh))
            
            h, w = scaled_template.shape[:2]
            for pt in zip(*locations[::-1]):
                confidence = 1 - result[pt[1], pt[0]]
                center_x = pt[0] + w // 2
                center_y = pt[1] + h // 2
                all_matches.append((confidence, center_x, center_y, w, h))
        
        if all_matches:
            all_matches = self._non_max_suppression(all_matches, min_distance)
        
        return [(conf, x, y) for conf, x, y, _, _ in all_matches]
    
    def _non_max_suppression(self, matches, min_distance):
        if not matches:
            return []
        
        matches = sorted(matches, key=lambda x: x[0], reverse=True)
        filtered = []
        
        for conf, x, y, w, h in matches:
            is_unique = True
            for f_conf, fx, fy, fw, fh in filtered:
                dx = abs(x - fx)
                dy = abs(y - fy)
                
                if dx < min_distance and dy < min_distance:
                    x1, y1 = max(x - w//2, fx - fw//2), max(y - h//2, fy - fh//2)
                    x2, y2 = min(x + w//2, fx + fw//2), min(y + h//2, fy + fh//2)
                    
                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = w * h
                        area2 = fw * fh
                        union = area1 + area2 - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.1:
                            is_unique = False
                            break
            
            if is_unique:
                filtered.append((conf, x, y, w, h))
        
        return filtered
