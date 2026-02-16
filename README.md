# 🕌 Islamic Reminder Video Bot

A Telegram bot that creates styled Islamic reminder videos (9:16) with Arabic audio, Albanian subtitles, circular video mask, and professional layout.

## Architecture

| Component | Tool | Cost |
|---|---|---|
| Bot Framework | python-telegram-bot 21.10 | Free |
| Video Processing | MoviePy 2.1 + FFmpeg | Free |
| Arabic Transcription | OpenAI Whisper (Local CPU) | Free |
| Translation Alignment | Google Gemini 2.5 Flash | Free Tier |
| Python | 3.12 (via Homebrew) | Free |

---

## Quick Start

```bash
cd selefi-org-ai-video-telegram-bot
source venv/bin/activate
python bot.py
```

Then open Telegram → find your bot → send `/start`.

---

## Setup (from scratch)

### Prerequisites

#### 1. Python 3.12
- **macOS:** `brew install python@3.12`
- **Check:** `python3.12 --version`

#### 2. FFmpeg
- **macOS:** `brew install ffmpeg`
- **Ubuntu:** `sudo apt install ffmpeg`

### API Keys

#### 🤖 Telegram Bot Token
1. Open Telegram → [@BotFather](https://t.me/BotFather)
2. Send `/newbot` → follow prompts → copy token

#### 🧠 Google Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in → Create API Key → copy it

### Install

```bash
# Create venv with Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# Install setuptools (needed for whisper) then all deps
pip install "setuptools<81" wheel
pip install openai-whisper==20240930 --no-build-isolation
pip install python-telegram-bot==21.10 moviepy==2.1.2 google-generativeai==0.8.4 python-dotenv==1.0.1

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Assets
Place your `logo.png` in the `assets/` folder. It will auto-scale.

---

## Running

```bash
source venv/bin/activate
python bot.py
```

The bot uses long-polling — no webhook or server required.

---

## Bot Flow

```
/start → Title → Author → Video File → Albanian Text → Confirm → Processing → Video Sent
```

1. User sends `/start`
2. Bot asks for **Title** text
3. Bot asks for **Author** name
4. Bot asks for **Video file** (Arabic audio)
5. Bot asks for **Albanian translation** text
6. User confirms → Bot processes:
   - Whisper transcribes Arabic → timestamps
   - Gemini aligns Albanian text to timestamps
   - MoviePy renders final 9:16 video
7. Bot sends the finished video back

---

## Video Layout

```
┌──────────────────────┐
│       [logo.png]     │  ← Centered, auto-scaled
│                      │
│  "Title Text Here"   │  ← Bold, wrapped, dark text
│                      │
│      ╭──────╮        │
│      │      │        │  ← Circular masked video
│      │ VIDEO│        │    with gold border
│      ╰──────╯        │
│                      │
│ ┌──────────────────┐ │  ← Semi-transparent subtitle box
│ │ Albanian text... │ │    synced to Arabic audio
│ └──────────────────┘ │
│                      │
│   "Author Name"      │  ← Gray, small text
└──────────────────────┘
      1080 × 1920
```

## Customization

Edit the **CONFIGURATION** block at the top of `video_engine.py`:

| Setting | Default | Description |
|---|---|---|
| `CANVAS_WIDTH/HEIGHT` | 1080×1920 | Output resolution |
| `BG_COLOR` | `#F5F5F5` | Background color |
| `CIRCLE_RADIUS` | 250 | Video circle size |
| `CIRCLE_CENTER_Y` | 850 | Circle vertical position |
| `CIRCLE_BORDER_COLOR` | `#D4AF37` | Gold border |
| `TITLE_FONT_SIZE` | 48 | Title text size |
| `SUBTITLE_FONT_SIZE` | 38 | Subtitle text size |
| `SUBTITLE_BG_COLOR` | `(0,0,0,180)` | Subtitle box opacity |
| `LOGO_MAX_WIDTH` | 300 | Logo max dimensions |
| `OUTPUT_BITRATE` | `4000k` | Video quality |

---

## Project Structure

```
selefi-org-ai-video-telegram-bot/
├── assets/            # logo.png goes here
├── temp/              # Temporary processing files (auto-cleaned)
├── output/            # Generated videos
├── .env               # Your API keys (not committed)
├── .env.example       # Template for .env
├── requirements.txt   # Python dependencies
├── setup.sh           # One-click setup script
├── transcriber.py     # Whisper + Gemini alignment logic
├── video_engine.py    # MoviePy video renderer + config
├── bot.py             # Telegram bot (main entry point)
├── test_phase2.py     # Test: transcription pipeline
└── README.md
```

## Phases

- [x] **Phase 1:** Environment & API Keys
- [x] **Phase 2:** Transcription & Sync Logic
- [x] **Phase 3:** Video Engine (MoviePy)
- [x] **Phase 4:** Telegram Bot Wrapper
