"""
Unit tests for src/main.py helper functions.

These tests avoid Streamlit UI and network calls:
- We only test pure helpers (size formatting, file extraction, TTS helpers).
- For file extraction, we load bytes into a BytesIO object and attach a `.name`
  attribute so main.extract_text_from_file() can branch by extension.
"""

import io
import sys
from pathlib import Path

# Ensure we import from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import main  # noqa: E402


def _bytesio_with_name(path: Path, fake_name: str) -> io.BytesIO:
    """
    Load file bytes into BytesIO and attach a `.name` attribute so the
    app's extractor can recognize the extension as if it came from an uploader.
    """
    data = path.read_bytes()
    bio = io.BytesIO(data)
    bio.name = fake_name  # real file objects' .name is read-only; BytesIO allows it
    return bio


def test_human_size_bytes():
    """human_size should format simple byte counts and not error."""
    assert main.human_size(500) == "500.0 B"
    assert main.human_size(2048).endswith("KB")


def test_tts_lang_normalization():
    """tts_lang should map 'zh' -> 'zh-cn' and pass through other codes."""
    assert main.tts_lang("zh") == "zh-cn"
    assert main.tts_lang("en") == "en"


def test_tts_supported_returns_bool():
    """tts_supported should return a boolean, not raise."""
    result = main.tts_supported("en")
    assert isinstance(result, bool)


def test_extract_text_from_txt():
    """TXT extraction should return string content, not raise."""
    path = ROOT / "testfiles" / "helloworld.txt"
    fobj = _bytesio_with_name(path, "helloworld.txt")
    text = main.extract_text_from_file(fobj)
    assert isinstance(text, str) and "hello world" in text.lower()


def test_extract_text_from_blank_file(tmp_path):
    """
    Blank TXT should return empty string or a friendly error message,
    but should not raise.
    """
    blank = tmp_path / "blankfile.txt"
    blank.write_text("", encoding="utf-8")
    fobj = _bytesio_with_name(blank, "blankfile.txt")
    text = main.extract_text_from_file(fobj)
    assert text == "" or "error" in text.lower()


def test_extract_text_from_pdf():
    """
    PDF extraction should include the known sentence from sample.pdf.
    (Some backends insert extra whitespace — we check the main phrase.)
    """
    path = ROOT / "testfiles" / "sample.pdf"
    fobj = _bytesio_with_name(path, "sample.pdf")
    text = main.extract_text_from_file(fobj)
    assert "Use this file to validate PDF extraction" in text


def test_extract_text_from_csv():
    """CSV extraction should render headers or values as text."""
    path = ROOT / "testfiles" / "sample.csv"
    fobj = _bytesio_with_name(path, "sample.csv")
    text = main.extract_text_from_file(fobj)
    assert "City" in text or "Name" in text
    assert "Alice" in text or "Bob" in text


def test_extract_text_from_png():
    """
    PNG OCR should return non-empty text if Tesseract is available and
    the image contains readable text.
    """
    path = ROOT / "testfiles" / "Capstone_png_image.png"
    fobj = _bytesio_with_name(path, "Capstone_png_image.png")
    text = main.extract_text_from_file(fobj)
    assert isinstance(text, str)
    assert text.strip() != ""


def test_extract_text_from_jpg():
    """
    JPG OCR should return non-empty text if Tesseract is available and
    the image contains readable text.
    """
    path = ROOT / "testfiles" / "Capstone_jpeg_image.jpg"
    fobj = _bytesio_with_name(path, "Capstone_jpeg_image.jpg")
    text = main.extract_text_from_file(fobj)
    assert isinstance(text, str)
    assert text.strip() != ""
