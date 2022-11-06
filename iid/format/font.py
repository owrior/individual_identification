from pathlib import Path

from PIL import ImageFont

GIDOLE_FONT = ImageFont.truetype(
    str(Path(__file__).parent / "Gidole-Regular.ttf"), size=22
)
