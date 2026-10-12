"""Render every chart in charts.html to PNG (1600x1000 CSS px at 2x) with Playwright.

  python assets/charts/render.py            # writes assets/charts/png/<chart>.png
"""

import glob
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
CHARTS = ["overview", "pareto", "size", "localllama", "adaptive", "latency", "finetune"]


def main():
    out = HERE / "png"
    out.mkdir(exist_ok=True)
    with sync_playwright() as p:
        # use any Chromium Playwright already downloaded when this package's pinned build is missing
        exe = os.environ.get("CHROMIUM") or next(iter(sorted(glob.glob(os.path.expanduser(
            "~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/*")))), None)
        try:
            browser = p.chromium.launch()
        except Exception:
            browser = p.chromium.launch(executable_path=exe)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        for name in CHARTS:
            page.goto(f"{(HERE / 'charts.html').as_uri()}#{name}")
            page.reload()  # a hash change alone does not re-run the page script
            page.wait_for_selector("body[data-ready='1']")
            page.wait_for_timeout(300)
            page.locator("#frame").screenshot(path=str(out / f"{name}.png"))
            print("wrote", out / f"{name}.png")
        browser.close()


if __name__ == "__main__":
    main()
