"""Generate synthetic test fixtures for HotGaze tests."""

from pathlib import Path

from PIL import Image, ImageDraw


def _fixture_dir() -> Path:
    d = Path(__file__).parent.parent / "tests" / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_landing_page() -> None:
    """Generate a synthetic landing-page-like image."""
    w, h = 800, 600
    img = Image.new("RGB", (w, h), (245, 245, 250))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, w, 60], fill=(30, 30, 80))
    draw.text((20, 18), "HotGaze", fill=(255, 255, 255))

    # Hero section
    draw.rectangle([100, 100, 700, 250], fill=(255, 255, 255), outline=(200, 200, 210))
    draw.text((150, 130), "Predict Visual Attention", fill=(30, 30, 80))
    draw.text((150, 160), "Open-source. Scriptable. CI-ready.", fill=(100, 100, 120))

    # CTA button
    draw.rectangle([250, 200, 450, 235], fill=(60, 120, 240))
    draw.text((280, 208), "Get Started", fill=(255, 255, 255))

    # Features section
    for i, feat in enumerate(["Fast heuristic engine", "Region scoring", "A/B comparison"]):
        y = 300 + i * 30
        draw.rectangle([100, y, 700, y + 25], fill=(255, 255, 255), outline=(220, 220, 230))
        draw.text((120, y + 5), feat, fill=(60, 60, 80))

    # Footer
    draw.rectangle([0, h - 40, w, h], fill=(30, 30, 80))
    draw.text((20, h - 30), "© 2026 HotGaze", fill=(180, 180, 200))

    img.save(_fixture_dir() / "landing.png")
    print("Created landing.png")


def make_landing_variant() -> None:
    """Generate a variant of the landing page with a different CTA position."""
    w, h = 800, 600
    img = Image.new("RGB", (w, h), (245, 245, 250))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, w, 60], fill=(30, 30, 80))
    draw.text((20, 18), "HotGaze Pro", fill=(255, 255, 255))

    draw.rectangle([100, 100, 700, 250], fill=(255, 255, 255), outline=(200, 200, 210))
    draw.text((150, 130), "Predict Visual Attention", fill=(30, 30, 80))
    draw.text((150, 160), "Now with A/B comparison", fill=(100, 100, 120))

    # CTA moved to top-right of hero
    draw.rectangle([550, 110, 680, 145], fill=(240, 60, 60))
    draw.text((565, 118), "Sign Up", fill=(255, 255, 255))

    img.save(_fixture_dir() / "landing_variant.png")
    print("Created landing_variant.png")


def make_1440x900() -> None:
    """Generate a 1440x900 test image for perf testing."""
    w, h = 1440, 900
    img = Image.new("RGB", (w, h), (240, 240, 245))
    draw = ImageDraw.Draw(img)

    # Simulate a complex UI
    draw.rectangle([0, 0, w, 50], fill=(25, 25, 70))
    for i in range(10):
        x = 50 + i * 140
        draw.rectangle([x, 80, x + 120, 200], fill=(255, 255, 255), outline=(200, 200, 210))
        draw.text((x + 10, 100), f"Card {i+1}", fill=(50, 50, 70))
    draw.rectangle([0, h - 40, w, h], fill=(25, 25, 70))

    img.save(_fixture_dir() / "1440x900.png")
    print("Created 1440x900.png")


if __name__ == "__main__":
    make_landing_page()
    make_landing_variant()
    make_1440x900()
    print("All fixtures generated.")
