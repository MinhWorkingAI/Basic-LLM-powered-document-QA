"""
utils.py — shared utilities for main-cpu.py and main-gpu.py

Responsibilities:
  - Load CPU_LLM / GPU_LLM from .env
  - Load generation config (TEMPERATURE, MAX_NEW_TOKENS_*) from .env
  - Auto-detect and read the single file inside /text (supports .txt and .pdf)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── .env ──────────────────────────────────────────────────────────────────────

load_dotenv()


def get_model_name(mode: str) -> str:
    """
    Return the model name for the given mode.

    Args:
        mode: "cpu" or "gpu"

    Returns:
        Model name string from .env (CPU_LLM or GPU_LLM)

    Raises:
        ValueError: if mode is invalid or the .env variable is not set
    """
    mode = mode.lower()
    if mode == "cpu":
        model = os.getenv("CPU_LLM", "").strip()
        env_key = "CPU_LLM"
    elif mode == "gpu":
        model = os.getenv("GPU_LLM", "").strip()
        env_key = "GPU_LLM"
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'cpu' or 'gpu'.")

    if not model:
        raise ValueError(
            f"Environment variable '{env_key}' is not set or is empty. "
            f"Please check your .env file."
        )

    return model


def get_generation_config() -> dict:
    """
    Load generation parameters from .env with sensible defaults.

    Returns:
        dict with keys: temperature, max_new_tokens_summary, max_new_tokens_qa
    """
    temperature          = float(os.getenv("TEMPERATURE", "0.7"))
    max_new_tokens_summary = int(os.getenv("MAX_NEW_TOKENS_SUMMARY", "300"))
    max_new_tokens_qa    = int(os.getenv("MAX_NEW_TOKENS_QA", "300"))

    return {
        "temperature":           temperature,
        "max_new_tokens_summary": max_new_tokens_summary,
        "max_new_tokens_qa":     max_new_tokens_qa,
    }


# ── /text file reader ─────────────────────────────────────────────────────────

TEXT_DIR = Path("text")


def read_text_file() -> str:
    """
    Auto-detect and read the single file inside the /text directory.
    Supports .txt and .pdf files.

    Returns:
        The full text content as a string.

    Raises:
        FileNotFoundError: if /text directory doesn't exist or is empty
        ValueError: if more than one file is found, or file type is unsupported
    """
    if not TEXT_DIR.exists():
        raise FileNotFoundError(
            f"Directory '{TEXT_DIR}' does not exist. "
            f"Please create a 'text/' folder and place your document inside."
        )

    # Collect all files (ignore hidden files like .DS_Store)
    files = [f for f in TEXT_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found in '{TEXT_DIR}'. Please add a .txt or .pdf file."
        )

    if len(files) > 1:
        names = [f.name for f in files]
        raise ValueError(
            f"Expected exactly one file in '{TEXT_DIR}', but found {len(files)}: {names}. "
            f"Please keep only one document."
        )

    file_path = files[0]
    suffix = file_path.suffix.lower()

    print(f"[utils] Reading document: {file_path.name}")

    if suffix == ".txt":
        return _read_txt(file_path)
    elif suffix == ".pdf":
        return _read_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Only .txt and .pdf are supported."
        )


def _read_txt(path: Path) -> str:
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"File '{path.name}' is empty.")
    return content


def _read_pdf(path: Path) -> str:
    """Read a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required to read PDF files. Install it with: pip install pypdf"
        )

    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())

    content = "\n\n".join(pages_text).strip()

    if not content:
        raise ValueError(
            f"Could not extract text from '{path.name}'. "
            f"The PDF may be scanned or image-based."
        )

    return content
