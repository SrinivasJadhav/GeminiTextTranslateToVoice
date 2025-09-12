"""
AI Translate & Voice (Streamlit)
--------------------------------
Translate user-provided text or documents to a target language and
generate natural-sounding audio (gTTS). The app:
- Extracts text from TXT/CSV/XLSX/PDF/Images (OCR via Tesseract).
- Translates using Gemini; falls back to deep_translator/googletrans if available.
- Generates MP3 audio from the translated text.
- Keeps the selected language stable through reruns.
- Uses external styles.css for UI polish.

Notes:
- We keep try/except inside all helpers so the UI never crashes.
- The output textarea is read-only by design.
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import gtts as gTTS
from gtts.lang import tts_langs
from dotenv import load_dotenv

# ======================
# ⚙️ Configuration
# ======================
# Set Tesseract executable path (update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Prefer pypdf; fallback to PyPDF2
try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader  # pragma: no cover

# Optional fallback translators (installed by user)
FALLBACK_AVAILABLE = False
try:
    from deep_translator import GoogleTranslator as DTGoogleTranslator
    FALLBACK_AVAILABLE = True
except Exception:
    try:
        from googletrans import Translator as GTTranslator  # googletrans==4.0.0rc1
        FALLBACK_AVAILABLE = True
    except Exception:
        FALLBACK_AVAILABLE = False

# Lazily initialized Gemini model
_genai_model = None


# -----------------------------
# Styling (external stylesheet)
# -----------------------------
def load_css(filename: str = "styles.css") -> None:
    """Load a CSS file and inject into the Streamlit page."""
    try:
        css_path = Path(__file__).parent / filename
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS file '{filename}': {e}")


# -----------------------------
# Utility helpers
# -----------------------------
def human_size(n: int) -> str:
    """Return a human-readable size string (e.g., '1.0 KB')."""
    try:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"
    except Exception as e:
        return f"Error: {e}"


def text_to_speech_gtts(text: str, lang: str = "en", slow: bool = False) -> str:
    """
    Convert text to speech (MP3) using gTTS.
    Returns path to a temporary .mp3 file.
    """
    try:
        tts = gTTS.gTTS(text=text, lang=lang, slow=slow)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        # Raise to the UI handler so it can show a friendly message
        raise RuntimeError(f"TTS failed: {e}") from e


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from a PDF-like stream using pypdf/PyPDF2.
    Returns an empty string if nothing is extractable.
    """
    try:
        reader = PdfReader(uploaded_file)
        out = []
        for page in getattr(reader, "pages", []):
            try:
                t = page.extract_text()
            except Exception:
                t = ""
            if t:
                out.append(t)
        return "\n".join(out).strip()
    except Exception as e:
        return f"Error extracting PDF text: {e}"


def read_image_text(uploaded_file) -> str:
    """
    OCR an image stream using Tesseract via pytesseract.
    Returns recognized text, or an error string (never raises).
    """
    try:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes))
        # If Tesseract is missing, this may still fail; we catch below.
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            pass
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"Error extracting text from image: {e}"


def extract_text_from_file(uploaded_file) -> str:
    """
    Extract text from a file-like object by inspecting its filename extension.
    Supports: .pdf, .txt, .csv, .xlsx, .jpeg/.jpg/.png (OCR).
    Returns an empty string on unknown types or an error string on failures.
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".pdf"):
            return extract_text_from_pdf(uploaded_file)
        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore").strip()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file).to_string(index=False)
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file).to_string(index=False)
        if name.endswith((".jpeg", ".jpg", ".png")):
            return read_image_text(uploaded_file)
        return ""
    except Exception as e:
        return f"Error reading file: {e}"


def tts_lang(code: str) -> str:
    """Normalize language code for gTTS (e.g., 'zh' -> 'zh-cn')."""
    try:
        return "zh-cn" if code == "zh" else code
    except Exception:
        return code or "en"


def tts_supported(code: str) -> bool:
    """
    Return True if gTTS reports the code is supported.
    We keep this permissive; if the lookup fails, return True and let TTS call decide.
    """
    try:
        return tts_lang(code) in tts_langs().keys()
    except Exception:
        return True


# -----------------------------
# Gemini helpers (lazy)
# -----------------------------
def _get_gemini_model():
    """
    Create/cache a Gemini model instance using GOOGLE_API_KEY.
    Raises RuntimeError if the key is missing or init fails.
    """
    try:
        global _genai_model
        if _genai_model is not None:
            return _genai_model

        import google.generativeai as genai  # local import to avoid import cost on tests

        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set. Cannot use Gemini.")
        genai.configure(api_key=api_key)
        _genai_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        return _genai_model
    except Exception as e:
        raise RuntimeError(f"Gemini init failed: {e}") from e


def detect_language_gemini(text: str) -> str:
    """
    Heuristically detect ISO 639-1 code for a snippet using Gemini.
    Returns a 2-letter code (best effort) or empty string on error.
    """
    try:
        text = (text or "").strip()
        if not text:
            return ""
        model = _get_gemini_model()
        prompt = (
            "Detect the ISO 639-1 two-letter language code of the following text. "
            "Return ONLY the code (e.g., en, es, fr). No explanation.\n\n"
            f"Text:\n{text[:2000]}"
        )
        response = model.generate_content(prompt)
        code = (getattr(response, "text", "") or "").strip().lower()
        return code[:2]
    except Exception:
        return ""


def translate_text_gemini(text: str, target_language_name: str, target_language_code: str) -> Tuple[str, Optional[str]]:
    """
    Translate `text` to the given target language using Gemini.
    Returns (translated_text, error_message_or_None).
    """
    try:
        prompt = (
            f"Translate the text to {target_language_name} ({target_language_code}). "
            "Return ONLY the translated text in the target language. "
            "No labels, explanations, transliterations, or quotes.\n\nText:\n"
            f"{text}"
        )
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        out = (getattr(response, "text", "") or "").strip()
        if out:
            return out, None
        return "", "Gemini returned empty text."
    except Exception as e:
        return "", f"{e}"


def translate_text_fallback(text: str, target_code: str) -> Tuple[str, Optional[str]]:
    """
    Optional fallback translation via deep_translator or googletrans (if installed).
    Returns (translated_text, error_message_or_None).
    """
    try:
        if not FALLBACK_AVAILABLE:
            return "", "No fallback translator installed (deep-translator or googletrans)."
        # Try deep_translator first
        if "DTGoogleTranslator" in globals():
            try:
                tgt = "zh-CN" if target_code == "zh" else target_code
                out = DTGoogleTranslator(source="auto", target=tgt).translate(text)
                return (out or "").strip(), None
            except Exception:
                pass
        # Try googletrans second
        if "GTTranslator" in globals():
            try:
                tgt = "zh-cn" if target_code == "zh" else target_code
                t = GTTranslator()
                res = t.translate(text, dest=tgt)
                return (res.text or "").strip(), None
            except Exception:
                pass
        return "", "Fallback failed."
    except Exception as e:
        return "", f"Fallback error: {e}"


# -----------------------------
# Session defaults & audio gen
# -----------------------------
def _ensure_session_defaults() -> None:
    """Ensure essential keys exist in Streamlit session_state."""
    try:
        defaults = {
            "translated_text": "",
            "audio_file": None,
            "source_text": "",
            "last_detected_code": "",
            "translation_done": False,
            "target_language": "English",  # set once; do not overwrite later
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
    except Exception as e:
        st.warning(f"Session init issue: {e}")


def _clear_outputs() -> None:
    """Clear translated text + audio when language changes or user requests reset."""
    try:
        st.session_state["translated_text"] = ""
        st.session_state["translation_done"] = False
        af = st.session_state.get("audio_file")
        if af and os.path.exists(af):
            try:
                os.remove(af)
            except Exception:
                pass
        st.session_state["audio_file"] = None
    except Exception as e:
        st.warning(f"Could not clear outputs: {e}")


def _make_audio_from_session_text(tts_code: str, slow: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Generate/replace an MP3 file from st.session_state['translated_text'].
    Returns (ok, error_message_or_None).
    """
    try:
        # Remove previous audio if present
        af = st.session_state.get("audio_file")
        if af and os.path.exists(af):
            try:
                os.remove(af)
            except Exception:
                pass
            st.session_state["audio_file"] = None

        base_text = (st.session_state.get("translated_text") or "").strip()
        if not base_text:
            return False, "No translated text to synthesize."

        if not tts_supported(tts_code):
            return False, "Selected language is not supported for audio."

        audio_path = text_to_speech_gtts(base_text, lang=tts_code, slow=slow)
        st.session_state["audio_file"] = audio_path
        return True, None
    except Exception as e:
        return False, f"Audio generation failed: {e}"


# -----------------------------
# UI
# -----------------------------
def run_app() -> None:
    """Build and run the Streamlit UI."""
    try:
        APP_TITLE = "AI Translate & Voice"
        st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🌐")

        load_css("styles.css")

        st.markdown(
            f"<div class='app-title'>🌐 {APP_TITLE}</div>"
            "<p class='app-subtitle'>Translate documents or text into a target language and generate natural audio. Clean, simple, fast.</p>",
            unsafe_allow_html=True,
        )

        # Optional: set Tesseract path via .env (TESSERACT_PATH)
        try:
            tess_path = os.getenv("TESSERACT_PATH")
            if tess_path and os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path
            _ = pytesseract.get_tesseract_version()
        except Exception:
            # No crash if OCR is missing—image extraction will return error strings
            pass

        _ensure_session_defaults()

        # ------------- Layout
        left, right = st.columns([1.05, 0.95], gap="large")

        # === Left: Input & Controls
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📝 Input</div>', unsafe_allow_html=True)

            st.sidebar.header("Input Options")
            input_method = st.sidebar.radio("Choose input method:", ("Text Input", "File Upload"))

            uploaded_file = None
            text = ""

            if input_method == "Text Input":
                # Use a non-empty label but hide visually for accessibility compliance
                user_text = st.text_area(
                    "Source Text",
                    height=180,
                    placeholder="Paste or type your text here…",
                    label_visibility="collapsed",
                )
                text = user_text or ""
                if text:
                    words = len(text.split())
                    chars = len(text)
                    st.markdown(
                        f'<div class="pills"><span class="pill">🗒 {words} words</span>'
                        f'<span class="pill">🔤 {chars} chars</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                uploaded_file = st.file_uploader(
                    "Upload a file (PDF, TXT, CSV, XLSX, JPEG, PNG)",
                    type=["pdf", "txt", "csv", "xlsx", "jpeg", "png"],
                )
                if uploaded_file is not None:
                    size = getattr(uploaded_file, "size", 0)
                    st.markdown(f'<span class="badge">File size: {human_size(size)}</span>', unsafe_allow_html=True)

                    # Guard rails: empty or >200MB files should stop early
                    max_bytes = 200 * 1024 * 1024
                    if size == 0:
                        st.error("Error: Empty file uploaded.")
                        st.stop()
                    if size > max_bytes:
                        st.error("Error: File size exceeds 200 MB limit.")
                        st.stop()

                    text = extract_text_from_file(uploaded_file)
                    if not text.strip():
                        st.error(
                            "This file appears to contain no extractable text. "
                            "If it’s a scanned PDF, OCR is required (convert pages to images and run OCR)."
                        )

            st.session_state["source_text"] = text

            # Nudge the user when there is no input
            no_input = (not text.strip()) and (uploaded_file is None)
            if no_input:
                st.markdown(
                    '<div class="empty-state">'
                    '<h4>👋 Start by adding content</h4>'
                    'Type or paste text above, <b>or</b> upload a PDF/TXT/CSV/XLSX/JPEG/PNG.'
                    '<br/>Then click <b>Translate</b> to get both text and audio.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="soft">', unsafe_allow_html=True)

            # Language selection that does NOT reset on rerun
            st.markdown('<div class="section-title">🌍 Target language</div>', unsafe_allow_html=True)

            language_codes = {
                "Arabic": "ar", "Chinese (Simplified)": "zh", "English": "en",
                "French": "fr", "German": "de", "Hindi": "hi", "Italian": "it",
                "Japanese": "ja", "Korean": "ko", "Russian": "ru", "Spanish": "es",
                "Tamil": "ta", "Telugu": "te",
            }
            options = list(language_codes.keys())

            # Use our stored selection (default to English once)
            current = st.session_state.get("target_language", "English")
            if current not in options:
                current = "English"
                st.session_state["target_language"] = current

            # Render WITHOUT a key; compute index from current value
            idx = options.index(current)
            selected = st.selectbox(
                "Target language",
                options,
                index=idx,
                label_visibility="collapsed",
            )

            # If user changed language, clear outputs (text + audio)
            if selected != st.session_state["target_language"]:
                st.session_state["target_language"] = selected
                _clear_outputs()

            target_language = st.session_state["target_language"]
            target_code = language_codes.get(target_language, "en")
            tts_code = tts_lang(target_code)

            has_input = not no_input
            c1, c2 = st.columns([1, 1])
            with c1:
                translate_clicked = st.button("🪄 Translate", use_container_width=True, disabled=not has_input)
            with c2:
                slow_voice = st.toggle("Slow voice", value=False, help="Speak slower when generating audio.")

            st.markdown("</div>", unsafe_allow_html=True)

        # === Right: Output
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📤 Output</div>', unsafe_allow_html=True)

            if no_input and not st.session_state.get("translated_text") and not st.session_state.get("audio_file"):
                st.markdown(
                    '<div class="empty-state"><h4>Nothing to show yet</h4>'
                    "Add text or upload a file on the left, then click <b>Translate</b>.</div>",
                    unsafe_allow_html=True,
                )

            # Translate & immediately generate audio
            if translate_clicked and has_input:
                try:
                    # Clear the read-only output box first for a clean refresh
                    st.session_state["display_text"] = ""

                    if not st.session_state["source_text"].strip():
                        st.warning("Please provide some text or upload a file.")
                    else:
                        with st.spinner("Translating…"):
                            st.session_state["translation_done"] = False

                            # 1) Gemini
                            tx, err = translate_text_gemini(
                                st.session_state["source_text"], target_language, target_code
                            )

                            # 2) Fallback (if needed)
                            if not tx and err:
                                st.warning(f"Gemini translation issue: {err}")
                                ftx, ferr = translate_text_fallback(st.session_state["source_text"], target_code)
                                if ftx:
                                    st.session_state["translated_text"] = ftx
                                    st.session_state["translation_done"] = True
                                    st.session_state["display_text"] = ftx
                                    st.success("Translation complete (fallback).")
                                else:
                                    st.error(f"Translation failed. {f'Fallback error: {ferr}' if ferr else ''}")
                                    st.session_state["translation_done"] = False
                            else:
                                if tx:
                                    st.session_state["translated_text"] = tx
                                    st.session_state["translation_done"] = True
                                    st.session_state["display_text"] = tx
                                    st.success("Translation complete.")
                                else:
                                    st.error("Translation failed or returned empty text.")
                                    st.session_state["translation_done"] = False
                except Exception as e:
                    st.error(f"Unexpected error during translation: {e}")

                # Generate audio right after a successful translation
                if st.session_state.get("translation_done"):
                    ok, audio_err = _make_audio_from_session_text(tts_code, slow=slow_voice)
                    if ok:
                        st.success("Audio generated.")
                    else:
                        st.warning(audio_err or "Could not generate audio.")

            # Show translated text (read-only)
            final_text = st.session_state.get("translated_text", "").strip()
            if final_text:
                st.caption(f"Translated Text ({target_language})")
                st.text_area(
                    "Translated Text",
                    value=final_text,
                    height=220,
                    key="display_text",
                    label_visibility="collapsed",
                    disabled=True,  # Make read-only
                )

                d1, _, d3 = st.columns([0.4, 0.3, 0.3])
                with d1:
                    try:
                        st.download_button(
                            "⬇️ Download text",
                            data=final_text.encode("utf-8"),
                            file_name=f"translation_{target_code}.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"Could not prepare text download: {e}")
                with d3:
                    st.markdown(
                        "<div class='pills' style='justify-content:flex-end'>"
                        f"<span class='pill'>🗒 {len(final_text.split())} words</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            # Show audio player + download if audio exists
            if st.session_state.get("audio_file"):
                try:
                    st.audio(st.session_state["audio_file"], format="audio/mp3")
                    a1, a2 = st.columns([0.55, 0.45])
                    with a1:
                        st.download_button(
                            "⬇️ Download Audio",
                            data=open(st.session_state["audio_file"], "rb"),
                            file_name=f"translated_audio_{target_code}.mp3",
                            mime="audio/mp3",
                            use_container_width=True,
                        )
                    with a2:
                        st.markdown(
                            "<div class='pills' style='justify-content:flex-end'>"
                            f"<span class='pill'>🎧 {target_language} ({tts_code})</span>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.warning(f"Audio display failed: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown(
            "<div class='footer-note'>Made with ❤️  •  "
            f"{datetime.now().strftime('%b %d, %Y')}  •  Translate = text + audio</div>",
            unsafe_allow_html=True,
        )

    except Exception as e:
        # Last-resort UI error handler
        st.error(f"App error: {e}")


if __name__ == "__main__":
    run_app()
