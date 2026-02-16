"""
Phase 4: Telegram Bot — Islamic Reminder Video Generator
=========================================================
Conversational flow:
  /start → Title → Author → Video File → Albanian Text → Processing → Send Video

Usage:
    python bot.py
"""

import os
import sys
import json
import uuid
import logging
import traceback
import atexit
import signal
from dotenv import load_dotenv

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Local modules
from transcriber import transcribe_and_translate, build_subtitles_from_decisions
from video_engine import render_video, TEMP_DIR, OUTPUT_DIR

# Load environment
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

if not TELEGRAM_BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN not set in .env")
    sys.exit(1)

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Conversation states
STATE_TITLE = 0
STATE_AUTHOR = 1
STATE_VIDEO = 2
STATE_SEQ_REVIEW = 3  # Interactive sequence-by-sequence review

# Max file sizes
MAX_VIDEO_SIZE_MB = 20  # Telegram Bot API getFile limit is 20 MB
MAX_OUTPUT_SIZE_MB = 50  # Telegram bot API allows up to 50 MB uploads

# PID file to prevent multiple instances
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot.pid")


# ==============================================================
# /start — Entry point
# ==============================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Greets the user and asks for the video title."""
    # Reset any previous session data
    context.user_data.clear()

    await update.message.reply_text(
        "🕌 *AudioSelefi Video Gjenerues*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Unë do të krijoj një video 9:16 me:\n"
        "• Logon tënde në krye\n"
        "• Titullin\n"
        "• Video rrethore në qendër\n"
        "• Titra shqip të sinkronizuara\n"
        "• Emrin e autorit në fund\n\n"
        "Le të fillojmë! 👇\n\n"
        "*Hapi 1/3:* Dërgomë *Titullin* e videos.\n"
        "_(Shembull: Përkujtim nga Kurani Famëlartë)_",
        parse_mode="Markdown",
    )
    return STATE_TITLE


# ==============================================================
# Step 1: Receive Title
# ==============================================================
async def receive_title(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    title = update.message.text.strip()
    if len(title) < 2:
        await update.message.reply_text("⚠️ Titulli është shumë i shkurtër. Dërgo një titull të plotë.")
        return STATE_TITLE
    if len(title) > 200:
        await update.message.reply_text("⚠️ Titulli është shumë i gjatë (max 200 shkronja). Provo përsëri.")
        return STATE_TITLE

    context.user_data["title"] = title
    await update.message.reply_text(
        f"✅ Titulli: *{title}*\n\n"
        f"*Hapi 2/3:* Tani dërgomë *Emrin e Autorit*.\n"
        f"_(Shembull: Shejkh Abdul-Aziz ibn Baz)_",
        parse_mode="Markdown",
    )
    return STATE_AUTHOR


# ==============================================================
# Step 2: Receive Author
# ==============================================================
async def receive_author(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    author = update.message.text.strip()
    if len(author) < 2:
        await update.message.reply_text("⚠️ Emri i autorit është shumë i shkurtër. Provo përsëri.")
        return STATE_AUTHOR
    if len(author) > 100:
        await update.message.reply_text("⚠️ Emri i autorit është shumë i gjatë (max 100 shkronja). Provo përsëri.")
        return STATE_AUTHOR

    context.user_data["author"] = author
    await update.message.reply_text(
        f"✅ Autori: *{author}*\n\n"
        f"*Hapi 3/3:* Tani dërgomë *Videon* 🎥\n"
        f"_(Video me audio arabisht. Max {MAX_VIDEO_SIZE_MB} MB)_\n\n"
        f"⚠️ Dërgoje si *dokument/file*, jo si video të kompresuar.",
        parse_mode="Markdown",
    )
    return STATE_VIDEO


# ==============================================================
# Step 3: Receive Video File
# ==============================================================
async def receive_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles both document and video message types."""
    # Get the file object — could be sent as document or video
    if update.message.document:
        file_obj = update.message.document
        file_name = file_obj.file_name or "video.mp4"
    elif update.message.video:
        file_obj = update.message.video
        file_name = f"video_{uuid.uuid4().hex[:8]}.mp4"
    else:
        await update.message.reply_text(
            "⚠️ Të lutem dërgo një skedar video. Nuk dallova asnjë skedar në mesazhin tënd."
        )
        return STATE_VIDEO

    # Check file size
    if file_obj.file_size and file_obj.file_size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(
            f"⚠️ Skedari është shumë i madh ({file_obj.file_size / (1024*1024):.1f} MB). "
            f"Maksimumi: {MAX_VIDEO_SIZE_MB} MB. Kompresoje dhe dërgoje përsëri."
        )
        return STATE_VIDEO

    # Download to temp
    await update.message.reply_text("⬇️ Duke shkarkuar videon...")
    session_id = uuid.uuid4().hex[:12]
    context.user_data["session_id"] = session_id

    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_path = os.path.join(session_dir, file_name)
    try:
        tg_file = await file_obj.get_file()
        await tg_file.download_to_drive(video_path)
    except Exception as e:
        if "file is too big" in str(e).lower() or "too big" in str(e).lower():
            await update.message.reply_text(
                f"⚠️ Skedari është shumë i madh për tu shkarkuar nga Telegram "
                f"({file_obj.file_size / (1024*1024):.1f} MB).\n\n"
                f"Telegram lejon shkarkimin deri në *{MAX_VIDEO_SIZE_MB} MB*.\n"
                f"Kompresoje videon ose dërgoje si *dokument/file* me madhësi më të vogël.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"⚠️ Gabim gjatë shkarkimit: {e}\nProvo përsëri."
            )
        return STATE_VIDEO

    context.user_data["video_path"] = video_path
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    await update.message.reply_text(
        f"✅ Video e pranuar: `{file_name}` ({file_size_mb:.1f} MB)\n\n"
        f"🔄 Duke përpunuar videon...\n"
        f"1️⃣ Transkriptimi i audios arabisht\n"
        f"2️⃣ Përkthimi i segmenteve me AI\n\n"
        f"⏳ Kjo mund të zgjasë disa minuta...",
        parse_mode="Markdown",
    )

    try:
        # Run transcription + AI translation
        ai_translations = transcribe_and_translate(video_path)

        # Store in user_data
        context.user_data["ai_translations"] = ai_translations
        context.user_data["user_decisions"] = {}  # idx -> text or None
        context.user_data["current_seq"] = 0

        total = len(ai_translations)
        await update.message.reply_text(
            f"✅ U gjetën *{total} sekuenca* nga audioja arabisht.\n\n"
            f"Tani do të shfaq çdo sekuencë një nga një.\n"
            f"Për secilën, ti mundesh:\n"
            f"• *Dërgo tekstin shqip* — përkthimi yt për atë sekuencë\n"
            f"• *Shtyp ⏭ Kalo* — kjo sekuencë nuk do të ketë titër\n\n"
            f"Le të fillojmë! 👇",
            parse_mode="Markdown",
        )

        # Send the first sequence
        await _send_sequence(update, context)
        return STATE_SEQ_REVIEW

    except Exception as e:
        logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
        await update.message.reply_text(
            f"❌ *Gabim gjatë transkriptimit:*\n"
            f"`{str(e)[:300]}`\n\n"
            f"Provo përsëri me /start.",
            parse_mode="Markdown",
        )
        _cleanup_session(context)
        return ConversationHandler.END


# ==============================================================
# HELPER: Send a sequence to user for review
# ==============================================================
async def _send_sequence(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the current sequence to the user with Skip button."""
    seq_idx = context.user_data["current_seq"]
    ai_translations = context.user_data["ai_translations"]
    total = len(ai_translations)
    seg = ai_translations[seq_idx]

    start = seg["start"]
    end = seg["end"]
    arabic = seg["arabic"]
    albanian_ai = seg["albanian_ai"]

    # Inline keyboard with Skip button
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("⏭ Kalo (pa titër)", callback_data=f"skip_{seq_idx}")]
    ])

    # Use the message object from either update.message or update.callback_query.message
    message = update.message if update.message else update.callback_query.message

    await message.reply_text(
        f"📌 *Sekuenca {seq_idx + 1}/{total}*\n"
        f"`[{start:.1f}s - {end:.1f}s]`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔵 AR: {arabic}\n\n"
        f"🟢 AI: _{albanian_ai}_\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📝 *Dërgo përkthimin tënd shqip* për këtë sekuencë,\n"
        f"ose shtyp *⏭ Kalo* për ta anashkaluar.",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


# ==============================================================
# HELPER: Advance to next sequence or render video
# ==============================================================
async def _advance_sequence(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Move to next sequence, or render video if all done."""
    ai_translations = context.user_data["ai_translations"]
    current = context.user_data["current_seq"]
    total = len(ai_translations)

    next_seq = current + 1
    if next_seq < total:
        context.user_data["current_seq"] = next_seq
        await _send_sequence(update, context)
        return STATE_SEQ_REVIEW
    else:
        # All sequences reviewed — render video
        return await _render_final_video(update, context)


# ==============================================================
# Step 4: Receive user's Albanian text for current sequence
# ==============================================================
async def receive_seq_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User typed their Albanian translation for the current sequence."""
    text = update.message.text.strip()
    seq_idx = context.user_data["current_seq"]
    total = len(context.user_data["ai_translations"])

    # Store user's text for this sequence
    context.user_data["user_decisions"][seq_idx] = text

    await update.message.reply_text(
        f"✅ Sekuenca {seq_idx + 1}/{total} — u ruajt.",
    )

    return await _advance_sequence(update, context)


# ==============================================================
# Step 4b: User tapped Skip button
# ==============================================================
async def handle_skip_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User pressed the Skip button for a sequence."""
    query = update.callback_query
    await query.answer()

    # Parse which sequence was skipped
    data = query.data  # e.g. "skip_3"
    seq_idx = int(data.split("_")[1])
    total = len(context.user_data["ai_translations"])

    # Store None = skipped
    context.user_data["user_decisions"][seq_idx] = None

    await query.edit_message_text(
        f"⏭ Sekuenca {seq_idx + 1}/{total} — u anashkalua (pa titër).",
    )

    # Make sure current_seq is in sync
    context.user_data["current_seq"] = seq_idx

    return await _advance_sequence(update, context)


# ==============================================================
# RENDER: Build subtitles & render final video
# ==============================================================
async def _render_final_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """All sequences reviewed. Build subtitles and render video."""
    message = update.message if update.message else update.callback_query.message

    user_decisions = context.user_data["user_decisions"]
    ai_translations = context.user_data["ai_translations"]
    title = context.user_data["title"]
    author = context.user_data["author"]
    video_path = context.user_data["video_path"]
    session_id = context.user_data["session_id"]

    # Count stats
    translated = sum(1 for v in user_decisions.values() if v is not None)
    skipped = sum(1 for v in user_decisions.values() if v is None)

    await message.reply_text(
        f"📋 *Përmbledhje:*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Të përkthyera: {translated} sekuenca\n"
        f"⏭ Të anashkaluara: {skipped} sekuenca\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🔄 Duke përpiluar videon finale...\n"
        f"⏳ Kjo mund të zgjasë disa minuta...",
        parse_mode="Markdown",
    )

    try:
        # Build subtitles from user decisions
        subtitles = build_subtitles_from_decisions(ai_translations, user_decisions)

        if not subtitles:
            await message.reply_text(
                "⚠️ Nuk ka asnjë titër — i ke anashkaluar të gjitha sekuencat.\n"
                "Dërgo /start për të filluar përsëri."
            )
            _cleanup_session(context)
            return ConversationHandler.END

        # Render video
        output_filename = f"islamic_reminder_{session_id}.mp4"
        output_path = render_video(
            video_path=video_path,
            title=title,
            author=author,
            subtitles=subtitles,
            output_filename=output_filename,
        )

        # Send the video back
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        await message.reply_text(
            f"📤 Duke ngarkuar videon ({file_size_mb:.1f} MB)..."
        )

        with open(output_path, "rb") as f:
            await message.reply_video(
                video=f,
                caption=(
                    f"🕌 *{title}*\n"
                    f"📖 {author}\n\n"
                    f"Gjeneruar nga Selefi.org AI Video Bot"
                ),
                parse_mode="Markdown",
                supports_streaming=True,
                read_timeout=300,
                write_timeout=300,
            )

        await message.reply_text(
            "✅ *Përfundoi!* Dërgo /start për të krijuar një video tjetër.",
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.error(f"Render error: {e}\n{traceback.format_exc()}")
        await message.reply_text(
            f"❌ *Gabim gjatë përpilimit:*\n"
            f"`{str(e)[:300]}`\n\n"
            f"Provo përsëri me /start.",
            parse_mode="Markdown",
        )

    finally:
        _cleanup_session(context)

    return ConversationHandler.END


# ==============================================================
# /cancel — Cancel at any point
# ==============================================================
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _cleanup_session(context)
    await update.message.reply_text(
        "❌ Anuluar. Dërgo /start për të filluar përsëri.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


# ==============================================================
# /help
# ==============================================================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🕌 *Bot i Videove Islame — Ndihmë*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "*Komandat:*\n"
        "/start — Fillo krijimin e një videoje\n"
        "/cancel — Anulo sesionin aktual\n"
        "/help — Shfaq këtë mesazh ndihme\n\n"
        "*Si funksionon:*\n"
        "1. Dërgo *Titullin* e videos\n"
        "2. Dërgo *Emrin e Autorit*\n"
        "3. Ngarko *Videon* me audio arabisht\n"
        "4. Boti transkripaton dhe përkthën secilën sekuencë\n"
        "5. Ti shqyrton çdo sekuencë — dërgo përkthimin ose kalo\n"
        "6. Boti gjeneron videon finale me titrat e tua!\n\n"
        "*Këshilla:*\n"
        "• Dërgoje videon si *dokument* për cilësi më të mirë\n"
        "• Mbaji videot nën 50 MB\n"
        "• Video më e shkurtër = përpunim më i shpejtë\n"
        "• Mund ta kalosh çdo sekuencë që nuk dëshiron ta titrosh",
        parse_mode="Markdown",
    )


# ==============================================================
# Cleanup temp files for a session
# ==============================================================
def _cleanup_session(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Removes temporary files created during this session."""
    session_id = context.user_data.get("session_id")
    if session_id:
        session_dir = os.path.join(TEMP_DIR, session_id)
        if os.path.exists(session_dir):
            import shutil
            try:
                shutil.rmtree(session_dir)
                logger.info(f"Cleaned up session: {session_id}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {session_id}: {e}")

    # Also clean output after a delay? No — user might want to re-download.
    context.user_data.clear()


# ==============================================================
# Error handler
# ==============================================================
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update {update} caused error: {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "❌ Ndodhi një gabim i papritur. Provo përsëri me /start."
        )


# ==============================================================
# Main
# ==============================================================
def _acquire_pid_lock():
    """Ensure only one bot instance runs at a time."""
    # Check if another instance is already running
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                old_pid = int(f.read().strip())
            # Check if that process is still alive
            os.kill(old_pid, 0)  # signal 0 = check existence
            print(f"⚠️  Another bot instance is already running (PID {old_pid}).")
            print(f"   Kill it first with: kill {old_pid}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            # Process is dead — stale PID file, safe to overwrite
            pass
        except PermissionError:
            print(f"⚠️  Another bot instance is running (PID exists but cannot check).")
            sys.exit(1)

    # Write our PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Cleanup on exit
    def _remove_pid_file():
        try:
            os.remove(PID_FILE)
        except OSError:
            pass

    atexit.register(_remove_pid_file)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def main():
    print("=" * 60)
    print("🕌 Islamic Reminder Video Bot — Starting")
    print("=" * 60)

    _acquire_pid_lock()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Conversation handler: the main flow
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            STATE_TITLE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_title),
            ],
            STATE_AUTHOR: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_author),
            ],
            STATE_VIDEO: [
                MessageHandler(
                    filters.Document.ALL | filters.VIDEO,
                    receive_video,
                ),
            ],
            STATE_SEQ_REVIEW: [
                CallbackQueryHandler(handle_skip_callback, pattern=r"^skip_\d+$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_seq_text),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        allow_reentry=True,
        per_message=False,
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_error_handler(error_handler)

    print("✅ Bot is running! Press Ctrl+C to stop.")
    print(f"   Open Telegram and search for your bot.")
    print(f"   Send /start to begin.")
    print("=" * 60)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
