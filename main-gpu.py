"""
main-gpu.py — Text Summarisation & Question Answering (GPU, 4-bit quantized)

Uses the model defined in GPU_LLM inside .env.
Automatically reads the single document from /text.

Compatible with any HuggingFace instruct/chat model that supports
tokenizer.apply_chat_template() — just update GPU_LLM in .env.

Requirements:
    pip install transformers torch accelerate bitsandbytes python-dotenv pypdf
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import get_model_name, get_generation_config, read_text_file

# ── Constants ─────────────────────────────────────────────────────────────────

DEVICE = "cuda"

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str):
    """
    Load tokenizer and model from HuggingFace Hub with 4-bit quantization.

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        (tokenizer, model) tuple
    """
    print(f"[gpu] Loading model: {model_name}")
    print("[gpu] Using 4-bit quantization (bitsandbytes)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,       # extra memory savings
        bnb_4bit_quant_type="nf4",            # recommended for LLMs
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,  # required for some models (e.g. Qwen)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",        # automatically places layers across GPUs/CPU
        trust_remote_code=True,
    )

    model.eval()
    print("[gpu] Model loaded successfully.\n")
    return tokenizer, model


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(tokenizer, model, messages: list, max_new_tokens: int, temperature: float) -> str:
    """
    Run a single inference pass using apply_chat_template.

    Args:
        tokenizer: loaded tokenizer
        model: loaded model
        messages: list of {"role": ..., "content": ...} dicts
        max_new_tokens: max tokens to generate
        temperature: sampling temperature (0.0 = deterministic, >0 = creative)

    Returns:
        Generated text string (assistant reply only)
    """
    # apply_chat_template works for Qwen, DeepSeek, Llama-instruct, Mistral, etc.
    # Qwen returns a BatchEncoding (dict-like) rather than a bare tensor,
    # so we explicitly unpack input_ids to ensure .shape works correctly.
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"] if not isinstance(tokenized, torch.Tensor) else tokenized
    input_ids = input_ids.to(DEVICE)

    # temperature=0 means greedy decoding (do_sample=False)
    do_sample = temperature > 0.0

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the input prompt)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Task functions ─────────────────────────────────────────────────────────────

def summarise(tokenizer, model, paragraph: str, gen_config: dict) -> str:
    """Generate a concise summary of the given paragraph."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Summarise the provided text concisely in 3-5 sentences. "
                "Do not add information that is not in the text."
            ),
        },
        {
            "role": "user",
            "content": f"Please summarise the following text:\n\n{paragraph}",
        },
    ]
    return run_inference(
        tokenizer, model, messages,
        max_new_tokens=gen_config["max_new_tokens_summary"],
        temperature=gen_config["temperature"],
    )


def answer_question(tokenizer, model, paragraph: str, question: str, gen_config: dict) -> str:
    """
    Answer a question grounded strictly to the provided paragraph.
    If the answer is not in the text, the model will say so.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions strictly based on the provided text. "
                "If the answer cannot be found in the text, say: "
                "'I cannot find the answer to that in the provided text.' "
                "Do not use any outside knowledge."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Text:\n{paragraph}\n\n"
                f"Question: {question}"
            ),
        },
    ]
    return run_inference(
        tokenizer, model, messages,
        max_new_tokens=gen_config["max_new_tokens_qa"],
        temperature=gen_config["temperature"],
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Text Summarisation & QA — GPU Mode (4-bit)")
    print("=" * 60)

    # 1. Load model name and generation config from .env
    model_name = get_model_name("gpu")
    gen_config = get_generation_config()

    print(f"[gpu] Temperature: {gen_config['temperature']}")
    print(f"[gpu] Max tokens (summary): {gen_config['max_new_tokens_summary']}")
    print(f"[gpu] Max tokens (QA):      {gen_config['max_new_tokens_qa']}\n")

    # 2. Read the document from /text
    paragraph = read_text_file()

    # 3. Load model
    tokenizer, model = load_model(model_name)

    # 4. Auto-generate summary on startup
    print("Generating summary...\n")
    summary = summarise(tokenizer, model, paragraph, gen_config)
    print("─" * 60)
    print("SUMMARY:")
    print(summary)
    print("─" * 60)

    # 5. QA loop
    print("\nYou can now ask questions about the document.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Enter question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Exiting. Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        print("\nAnswering...\n")
        answer = answer_question(tokenizer, model, paragraph, question, gen_config)
        print("─" * 60)
        print("ANSWER:")
        print(answer)
        print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
