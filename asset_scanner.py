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
        self._template_cache = {}

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

        if len(template_files) == 1:
            template_name, template_data = self._load_template(template_files[0])
            if template_data is not None:
                templates[template_name] = template_data
                logger.info(f"Loaded template: {template_name}")
        else:
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
        if required_set:
            required_lookup = {name.lower() for name in required_set}
        else:
            required_lookup = None

        template_files = []
        with os.scandir(assets_path) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.lower().endswith(".png"):
                    continue
                template_name = Path(name).stem
                if required_lookup and template_name.lower() not in required_lookup:
                    continue
                template_files.append(Path(entry.path))
        template_files.sort()
        return template_files

    def _load_template(self, template_file):
        template_name = template_file.stem
        try:
            mtime = template_file.stat().st_mtime
        except OSError:
            mtime = None

        cached = self._template_cache.get(str(template_file))
        if cached and cached["mtime"] == mtime:
            return template_name, cached["data"]

        template_img = self.image_matcher.load_template(template_file)
        self._template_cache[str(template_file)] = {"mtime": mtime, "data": template_img}
        return template_name, template_img
