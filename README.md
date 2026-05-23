# Text Summarisation & Question Answering

A CLI-based NLP application using pretrained LLMs from HuggingFace for text summarisation and document-grounded question answering.

---

## File Structure

```
project/
├── .env                  # Model names and generation config
├── utils.py              # Shared utilities (model loader, file reader)
├── main-gpu.py           # GPU inference (4-bit quantized)
├── main-cpu.py           # CPU inference (float32)
└── text/
    └── your_document.pdf / .txt
```

---

## Dependencies

```
transformers
torch
accelerate
bitsandbytes
python-dotenv
pypdf
```

Install:
```bash
pip install transformers torch accelerate bitsandbytes python-dotenv pypdf
```

---

## Configuration

Edit `.env` before running:

```env
CPU_LLM="Qwen/Qwen2.5-1.5B-Instruct"
GPU_LLM="Qwen/Qwen2.5-7B-Instruct"

TEMPERATURE=0.7
MAX_NEW_TOKENS_SUMMARY=300
MAX_NEW_TOKENS_QA=300
```

Any HuggingFace instruct/chat model works — no code changes needed, just update the model name.

---

## Usage

Place a single `.txt` or `.pdf` file inside the `text/` folder, then run:

```bash
# GPU (recommended)
python main-gpu.py

# CPU
python main-cpu.py
```

On startup, the app will:
1. Load the model from HuggingFace Hub
2. Auto-read the document in `text/`
3. Generate a summary
4. Enter a QA loop — type your question, or `quit` to exit

---

## Notes

- `text/` must contain exactly one file
- GPU mode requires a CUDA-compatible GPU with sufficient VRAM (~6GB+ for 7B 4-bit)
- CPU mode is functional but slow on larger models; use a 1.5B–3B model for reasonable speed
- QA answers are grounded strictly to the document — the model will say so if the answer isn't there

---

*Created with the support of Claude Sonnet 4.6*
