"""
Phase 3: Video Engine — MoviePy Layout Builder
===============================================
Renders the final 9:16 Islamic Reminder video with:
- Static background color
- Logo at the top
- Title text (bold, wrapped)
- Circular masked video in the center
- Synchronized Albanian subtitles (lower third)
- Author text at the bottom

ALL visual settings are defined as constants at the top.
Edit the CONFIGURATION block to change the layout without touching the code.

Usage (standalone test):
    python video_engine.py
"""

import os
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoFileClip,
    ImageClip,
    TextClip,
    CompositeVideoClip,
    AudioFileClip,
    VideoClip,
    concatenate_videoclips,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║               CONFIGURATION (EDIT HERE)                      ║
# ║  Change any value below to adjust the video layout.          ║
# ║  No need to touch any code below this block.                 ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Canvas ---
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
BG_COLOR = "#F5F5F5"           # Off-white background

# --- Logo ---
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
LOGO_MAX_WIDTH = 300           # Max width (auto-scaled keeping aspect ratio)
LOGO_MAX_HEIGHT = 300          # Max height
LOGO_Y_POSITION = 80          # Distance from top edge

# --- Title ---
TITLE_FONT_SIZE = 48
TITLE_FONT_COLOR = "#1A1A1A"  # Near-black
TITLE_FONT = "Helvetica-Neue-Bold"
TITLE_Y_POSITION = 420        # Distance from top edge
TITLE_MAX_WIDTH = 900          # Max text width before wrapping
TITLE_LINE_SPACING = 10       # Extra pixels between lines

# --- Video Circle ---
CIRCLE_RADIUS = 250            # Radius of the circular video mask
CIRCLE_CENTER_Y = 850          # Y position of circle center
CIRCLE_BORDER_COLOR = "#D4AF37" # Gold border color
CIRCLE_BORDER_WIDTH = 5        # Border thickness in pixels

# --- Subtitles ---
SUBTITLE_FONT_SIZE = 38
SUBTITLE_FONT_COLOR = "#364242"  # Dark teal
SUBTITLE_BG_COLOR = (0, 0, 0, 0)  # Fully transparent (no background)
SUBTITLE_FONT = "Helvetica-Neue-Bold"
SUBTITLE_Y_POSITION = 1350     # Distance from top edge
SUBTITLE_MAX_WIDTH = 950       # Max width before wrapping
SUBTITLE_PADDING_X = 40        # Horizontal padding inside the background box
SUBTITLE_PADDING_Y = 20        # Vertical padding inside the background box
SUBTITLE_BORDER_RADIUS = 15    # Rounded corner radius for subtitle background

# --- Author ---
AUTHOR_FONT_SIZE = 30
AUTHOR_FONT_COLOR = "#888888"  # Gray
AUTHOR_FONT = "Helvetica-Neue"
AUTHOR_Y_POSITION = 1780       # Distance from top edge

# --- Translator ---
TRANSLATOR_FONT_SIZE = 26
TRANSLATOR_FONT_COLOR = "#888888"  # Gray
TRANSLATOR_FONT = "Helvetica-Neue"
TRANSLATOR_Y_POSITION = 1740   # Above the author name
TRANSLATOR_SHOW_SECONDS = 3    # Show only in the last N seconds

# --- Output ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
OUTPUT_FPS = 24
OUTPUT_CODEC = "libx264"
OUTPUT_AUDIO_CODEC = "aac"
OUTPUT_BITRATE = "2500k"
TELEGRAM_MAX_SIZE_MB = 49  # Telegram limit is 50 MB, use 49 for safety

# ╔══════════════════════════════════════════════════════════════╗
# ║            END OF CONFIGURATION — CODE BELOW                 ║
# ╚══════════════════════════════════════════════════════════════╝

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ==============================================================
# HELPER: Find a usable font path on the system
# ==============================================================
_FONT_CACHE = {}

def _find_font(font_name: str, bold: bool = False) -> str:
    """Resolves a font name to an actual file path on macOS/Linux."""
    cache_key = (font_name, bold)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    # Common macOS font paths
    search_paths = [
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        # Assets folder for custom fonts
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
    ]

    # Map common names to filenames
    font_map = {
        "Helvetica-Neue-Bold": ["HelveticaNeue.ttc", "Helvetica Neue Bold.ttf", "HelveticaNeue-Bold.ttf"],
        "Helvetica-Neue": ["HelveticaNeue.ttc", "Helvetica Neue.ttf", "HelveticaNeue.ttf"],
        "Arial-Bold": ["Arial Bold.ttf", "ArialHB.ttc", "Arial.ttf"],
        "Arial": ["Arial.ttf", "ArialHB.ttc"],
    }

    candidates = font_map.get(font_name, [f"{font_name}.ttf", f"{font_name}.ttc"])

    for search_dir in search_paths:
        for candidate in candidates:
            path = os.path.join(search_dir, candidate)
            if os.path.exists(path):
                _FONT_CACHE[cache_key] = path
                return path

    # Fallback
    fallback = "/System/Library/Fonts/Helvetica.ttc"
    if os.path.exists(fallback):
        _FONT_CACHE[cache_key] = fallback
        return fallback

    raise FileNotFoundError(f"Could not find font: {font_name}")


# ==============================================================
# HELPER: Render text to an image (Pillow-based for full control)
# ==============================================================
def _render_text_image(
    text: str,
    font_name: str,
    font_size: int,
    font_color: str,
    max_width: int,
    line_spacing: int = 5,
    align: str = "center",
) -> np.ndarray:
    """
    Renders wrapped text to a transparent RGBA numpy array using Pillow.
    Returns an RGBA numpy array suitable for ImageClip.
    """
    font_path = _find_font(font_name)
    font = ImageFont.truetype(font_path, font_size)

    # Word-wrap the text
    avg_char_width = font_size * 0.55
    wrap_width = max(int(max_width / avg_char_width), 10)
    wrapped_lines = textwrap.wrap(text, width=wrap_width)

    if not wrapped_lines:
        wrapped_lines = [" "]

    # Measure each line
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    line_sizes = []
    for line in wrapped_lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_sizes.append((w, h))

    total_width = max(w for w, h in line_sizes)
    line_height = max(h for w, h in line_sizes) + line_spacing
    total_height = line_height * len(wrapped_lines)

    # Render
    img = Image.new("RGBA", (total_width + 10, total_height + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y_offset = 0
    for i, line in enumerate(wrapped_lines):
        w, h = line_sizes[i]
        if align == "center":
            x = (total_width + 10 - w) // 2
        else:
            x = 0
        draw.text((x, y_offset), line, font=font, fill=font_color)
        y_offset += line_height

    return np.array(img)


# ==============================================================
# HELPER: Render subtitle with background box
# ==============================================================
def _render_subtitle_image(
    text: str,
    font_name: str,
    font_size: int,
    font_color: str,
    bg_color: tuple,
    max_width: int,
    padding_x: int,
    padding_y: int,
    border_radius: int,
) -> np.ndarray:
    """Renders subtitle text with a rounded semi-transparent background."""
    font_path = _find_font(font_name)
    font = ImageFont.truetype(font_path, font_size)

    # Word-wrap
    avg_char_width = font_size * 0.52
    wrap_width = max(int(max_width / avg_char_width), 10)
    wrapped_lines = textwrap.wrap(text, width=wrap_width)

    if not wrapped_lines:
        wrapped_lines = [" "]

    # Enforce maximum 2 lines for subtitle readability on phone screens
    if len(wrapped_lines) > 2:
        wrapped_lines = wrapped_lines[:2]

    # Measure
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    line_sizes = []
    for line in wrapped_lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_sizes.append((w, h))

    text_width = max(w for w, h in line_sizes)
    line_height = max(h for w, h in line_sizes) + 8
    text_height = line_height * len(wrapped_lines)

    box_width = text_width + padding_x * 2
    box_height = text_height + padding_y * 2

    # Draw background box with rounded corners
    img = Image.new("RGBA", (box_width, box_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [(0, 0), (box_width - 1, box_height - 1)],
        radius=border_radius,
        fill=bg_color,
    )

    # Draw text centered in the box
    y_offset = padding_y
    for i, line in enumerate(wrapped_lines):
        w, h = line_sizes[i]
        x = (box_width - w) // 2
        draw.text((x, y_offset), line, font=font, fill=font_color)
        y_offset += line_height

    return np.array(img)


# ==============================================================
# HELPER: Create circular mask for the video
# ==============================================================
def _create_circle_mask(radius: int) -> np.ndarray:
    """Creates a circular alpha mask (grayscale, white circle on black)."""
    diameter = radius * 2
    mask = Image.new("L", (diameter, diameter), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([0, 0, diameter - 1, diameter - 1], fill=255)
    return np.array(mask) / 255.0


def _create_circle_border(radius: int, border_width: int, border_color: str) -> np.ndarray:
    """Creates a ring/border image for the circle."""
    diameter = radius * 2 + border_width * 2
    img = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Outer circle
    draw.ellipse(
        [0, 0, diameter - 1, diameter - 1],
        outline=border_color,
        width=border_width,
    )
    return np.array(img)


# ==============================================================
# HELPER: Prepare & mask the input video into a circle
# ==============================================================
def _prepare_circular_video(video_path: str) -> VideoFileClip:
    """
    Loads the input video, resizes it to fit the circle,
    and applies a circular mask.
    """
    video = VideoFileClip(video_path)

    # Resize to fit circle diameter
    diameter = CIRCLE_RADIUS * 2
    # Scale so the shorter dimension fills the circle
    vw, vh = video.size
    scale = max(diameter / vw, diameter / vh)
    new_w = int(vw * scale)
    new_h = int(vh * scale)

    video = video.resized((new_w, new_h))

    # Center-crop to exact circle diameter
    x_center = new_w // 2
    y_center = new_h // 2
    x1 = x_center - CIRCLE_RADIUS
    y1 = y_center - CIRCLE_RADIUS
    video = video.cropped(x1=x1, y1=y1, x2=x1 + diameter, y2=y1 + diameter)

    # Apply circular mask
    circle_mask = _create_circle_mask(CIRCLE_RADIUS)
    mask_clip = ImageClip(circle_mask, is_mask=True).with_duration(video.duration)
    video = video.with_mask(mask_clip)

    return video


# ==============================================================
# HELPER: Load & resize logo
# ==============================================================
def _load_logo(logo_path: str) -> ImageClip:
    """Loads logo.png and resizes to fit within max dimensions."""
    if not os.path.exists(logo_path):
        raise FileNotFoundError(
            f"Logo not found at: {logo_path}\n"
            f"Please place your logo.png in the assets/ folder."
        )

    logo_img = Image.open(logo_path).convert("RGBA")
    w, h = logo_img.size

    # Scale down if needed, preserving aspect ratio
    scale = min(LOGO_MAX_WIDTH / w, LOGO_MAX_HEIGHT / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        logo_img = logo_img.resize((new_w, new_h), Image.LANCZOS)

    return np.array(logo_img)


# ==============================================================
# MAIN: Build the final composed video
# ==============================================================
def render_video(
    video_path: str,
    title: str,
    author: str,
    translator: str = "",
    subtitles: list[dict] = None,
    output_filename: str = "output_video.mp4",
) -> str:
    """
    Renders the final 9:16 Islamic Reminder video.

    Args:
        video_path: Path to the input video file (with Arabic audio).
        title: Title text to display.
        author: Author name to display at the bottom.
        translator: Translator name, shown above author in the last 3 seconds.
        subtitles: List of {"start": float, "end": float, "text": str}.
        output_filename: Name of the output file.

    Returns:
        Path to the rendered output video.
    """
    if subtitles is None:
        subtitles = []
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    print(f"\n🎬 Rendering video...")
    print(f"   Canvas: {CANVAS_WIDTH}x{CANVAS_HEIGHT}")
    print(f"   Title: {title}")
    print(f"   Author: {author}")
    print(f"   Subtitles: {len(subtitles)} segments")

    # --- Load input video ---
    print("   [1/6] Loading input video...")
    source_video = VideoFileClip(video_path)
    duration = source_video.duration

    # --- Background ---
    print("   [2/6] Creating background...")
    bg_img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    bg_array = np.array(bg_img)
    bg_clip = ImageClip(bg_array).with_duration(duration)

    layers = [bg_clip]

    # --- Logo ---
    print("   [3/6] Adding logo...")
    try:
        logo_array = _load_logo(LOGO_PATH)
        logo_clip = (
            ImageClip(logo_array)
            .with_duration(duration)
            .with_position(("center", LOGO_Y_POSITION))
        )
        layers.append(logo_clip)
    except FileNotFoundError as e:
        print(f"   ⚠️  {e} — Skipping logo.")

    # --- Title ---
    print("   [4/6] Rendering title...")
    title_img = _render_text_image(
        text=title,
        font_name=TITLE_FONT,
        font_size=TITLE_FONT_SIZE,
        font_color=TITLE_FONT_COLOR,
        max_width=TITLE_MAX_WIDTH,
        line_spacing=TITLE_LINE_SPACING,
        align="center",
    )
    title_clip = (
        ImageClip(title_img)
        .with_duration(duration)
        .with_position(("center", TITLE_Y_POSITION))
    )
    layers.append(title_clip)

    # --- Circular Video ---
    print("   [5/6] Creating circular video...")
    circular_video = _prepare_circular_video(video_path)
    circle_x = (CANVAS_WIDTH - CIRCLE_RADIUS * 2) // 2
    circle_y = CIRCLE_CENTER_Y - CIRCLE_RADIUS
    circular_video = circular_video.with_position((circle_x, circle_y))
    layers.append(circular_video)

    # Circle border
    if CIRCLE_BORDER_WIDTH > 0:
        border_img = _create_circle_border(
            CIRCLE_RADIUS, CIRCLE_BORDER_WIDTH, CIRCLE_BORDER_COLOR
        )
        border_clip = (
            ImageClip(border_img)
            .with_duration(duration)
            .with_position((
                circle_x - CIRCLE_BORDER_WIDTH,
                circle_y - CIRCLE_BORDER_WIDTH,
            ))
        )
        layers.append(border_clip)

    # --- Subtitles ---
    print("   [6/6] Burning subtitles...")
    for i, sub in enumerate(subtitles):
        sub_img = _render_subtitle_image(
            text=sub["text"],
            font_name=SUBTITLE_FONT,
            font_size=SUBTITLE_FONT_SIZE,
            font_color=SUBTITLE_FONT_COLOR,
            bg_color=SUBTITLE_BG_COLOR,
            max_width=SUBTITLE_MAX_WIDTH,
            padding_x=SUBTITLE_PADDING_X,
            padding_y=SUBTITLE_PADDING_Y,
            border_radius=SUBTITLE_BORDER_RADIUS,
        )
        sub_clip = (
            ImageClip(sub_img)
            .with_start(sub["start"])
            .with_end(sub["end"])
            .with_position(("center", SUBTITLE_Y_POSITION))
        )
        layers.append(sub_clip)

    # --- Author ---
    author_img = _render_text_image(
        text=author,
        font_name=AUTHOR_FONT,
        font_size=AUTHOR_FONT_SIZE,
        font_color=AUTHOR_FONT_COLOR,
        max_width=TITLE_MAX_WIDTH,
        align="center",
    )
    author_clip = (
        ImageClip(author_img)
        .with_duration(duration)
        .with_position(("center", AUTHOR_Y_POSITION))
    )
    layers.append(author_clip)

    # --- Translator (shown in last 3 seconds above author) ---
    if translator:
        translator_label = f"Përktheu: {translator}"
        translator_img = _render_text_image(
            text=translator_label,
            font_name=TRANSLATOR_FONT,
            font_size=TRANSLATOR_FONT_SIZE,
            font_color=TRANSLATOR_FONT_COLOR,
            max_width=TITLE_MAX_WIDTH,
            align="center",
        )
        translator_start = max(0, duration - TRANSLATOR_SHOW_SECONDS)
        translator_clip = (
            ImageClip(translator_img)
            .with_start(translator_start)
            .with_end(duration)
            .with_position(("center", TRANSLATOR_Y_POSITION))
        )
        layers.append(translator_clip)

    # --- Compose all layers ---
    print("   🔨 Compositing all layers...")
    final = CompositeVideoClip(layers, size=(CANVAS_WIDTH, CANVAS_HEIGHT))

    # Attach original audio
    final = final.with_audio(source_video.audio)

    # --- Render ---
    print(f"   💾 Writing to: {output_path}")
    final.write_videofile(
        output_path,
        fps=OUTPUT_FPS,
        codec=OUTPUT_CODEC,
        audio_codec=OUTPUT_AUDIO_CODEC,
        bitrate=OUTPUT_BITRATE,
        logger="bar",
    )

    # Cleanup
    source_video.close()
    final.close()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Video rendered: {output_path} ({file_size_mb:.1f} MB)")

    # Auto-compress if over Telegram limit
    if file_size_mb > TELEGRAM_MAX_SIZE_MB:
        print(f"   ⚠️  File exceeds {TELEGRAM_MAX_SIZE_MB} MB — compressing...")
        output_path = compress_for_telegram(output_path)

    return output_path


def compress_for_telegram(input_path: str) -> str:
    """
    Re-encodes a video with FFmpeg to fit under Telegram's 50 MB limit.
    Calculates the exact bitrate needed based on video duration.
    """
    import subprocess
    import json as _json

    # Get duration via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", input_path
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    duration = float(_json.loads(probe.stdout)["format"]["duration"])

    # Calculate target bitrate: (target_size_bytes * 8) / duration_seconds
    target_bytes = TELEGRAM_MAX_SIZE_MB * 1024 * 1024
    # Reserve 128kbps for audio
    audio_bitrate = 128  # kbps
    audio_bits = audio_bitrate * 1000 * duration
    video_bitrate = int(((target_bytes * 8) - audio_bits) / duration / 1000)  # kbps
    video_bitrate = max(video_bitrate, 500)  # Floor at 500kbps

    compressed_path = input_path.replace(".mp4", "_compressed.mp4")

    print(f"   🗜️  Target: {video_bitrate}k video + {audio_bitrate}k audio")
    print(f"   🗜️  Duration: {duration:.1f}s → target {TELEGRAM_MAX_SIZE_MB} MB")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-b:v", f"{video_bitrate}k",
        "-maxrate", f"{int(video_bitrate * 1.5)}k",
        "-bufsize", f"{video_bitrate * 2}k",
        "-preset", "medium",
        "-c:a", "aac", "-b:a", f"{audio_bitrate}k",
        "-movflags", "+faststart",
        compressed_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
    print(f"   ✅ Compressed: {compressed_size:.1f} MB")

    # Replace original with compressed
    os.replace(compressed_path, input_path)
    return input_path


# ==============================================================
# CLI: Standalone test with dummy data
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎬 Phase 3: Video Engine — Standalone Test")
    print("=" * 60)

    # Create a minimal test video if none exists
    test_video = os.path.join(TEMP_DIR, "test_video.mp4")
    if not os.path.exists(test_video):
        print("Creating test video...")
        from moviepy import AudioClip

        def make_frame(t):
            # Gradient frame for visual interest
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = 100  # Red channel
            frame[:, :, 1] = 80   # Green channel
            frame[:, :, 2] = 60   # Blue channel
            return frame

        def make_audio(t):
            return np.sin(2 * np.pi * 440 * t).reshape(-1, 1)

        v = VideoClip(make_frame, duration=10.0)
        a = AudioClip(make_audio, duration=10.0, fps=44100)
        v = v.with_audio(a).with_fps(24)
        v.write_videofile(test_video, codec="libx264", audio_codec="aac", logger=None)
        print(f"✅ Test video created: {test_video}")

    # Mock subtitles
    test_subtitles = [
        {"start": 0.0, "end": 3.5, "text": "Me emrin e Allahut, të Gjithëmëshirshmit, Mëshiruesit."},
        {"start": 3.5, "end": 7.0, "text": "Falënderimi i takon Allahut, Zotit të botëve."},
        {"start": 7.0, "end": 10.0, "text": "Të Gjithëmëshirshmit, Mëshiruesit."},
    ]

    result = render_video(
        video_path=test_video,
        title="Përkujtim i Rëndësishëm nga Kurani Famëlartë",
        author="Shejkh Abdul-Aziz ibn Baz",
        subtitles=test_subtitles,
        output_filename="test_output.mp4",
    )

    print(f"\n🎉 Test complete! Check: {result}")
    print("Open it to see the layout.")
