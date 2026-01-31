import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


logger = logging.getLogger(__name__)


class AssetScanner:
    def __init__(self, image_matcher, max_workers=None):
        self.image_matcher = image_matcher
        cpu_count = os.cpu_count() or 1
        self.max_workers = max_workers or min(32, cpu_count + 4)

    def scan(self, assets_dir, required_templates=None):
        assets_path = Path(assets_dir)
        if not assets_path.exists():
            logger.error(f"Assets directory not found: {assets_path}")
            return {}

        required_set = set(required_templates or [])
        template_files = self._collect_template_files(assets_path, required_set)

        templates = {}
        if not template_files:
            return templates

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_template, template_file): template_file
                for template_file in template_files
            }
            for future in as_completed(futures):
                template_file = futures[future]
                try:
                    template_name, template_data = future.result()
                except Exception as exc:
                    logger.error(f"Failed to load template {template_file}: {exc}")
                    continue

                if template_data is None:
                    continue

                templates[template_name] = template_data
                logger.info(f"Loaded template: {template_name}")

        if required_set:
            missing = sorted(required_set.difference(templates.keys()))
            if missing:
                logger.warning(f"Missing {len(missing)} required templates: {', '.join(missing)}")

        return templates

    def _collect_template_files(self, assets_path, required_set):
        template_files = []
        for entry in os.scandir(assets_path):
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(".png"):
                continue
            template_name = Path(entry.name).stem
            if required_set and template_name not in required_set:
                continue
            template_files.append(Path(entry.path))
        return template_files

    def _load_template(self, template_file):
        template_name = template_file.stem
        template_img = self.image_matcher.load_template(template_file)
        return template_name, template_img
