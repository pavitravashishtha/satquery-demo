"""
test_interpreter_with_qwen.py — REAL integration of Person B's Qwen3.5-2B
with Person A's QueryInterpreter (interpreter.py / schema.py / query_bank.py).

This is the actual deliverable: it loads Qwen, wraps it as qwen_llm_fn,
passes that into QueryInterpreter, and runs every query in the labeled
query bank through the combined system — printing per-query OK/MISS,
the source field (llm / hybrid / keyword_fallback), and final accuracy.

Run from inside the `satquery` folder (same folder as interpreter.py,
schema.py, query_bank.py, prompt_template.py, keyword_fallback.py):

    python test_interpreter_with_qwen.py
    python test_interpreter_with_qwen.py --verbose        # print raw Qwen output on every MISS
    python test_interpreter_with_qwen.py --limit 20        # only run first 20 queries (fast smoke test)

Needs: pip install transformers accelerate bitsandbytes torch pydantic
"""

import argparse
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from interpreter import QueryInterpreter
from query_bank import get_query_bank
from schema import Source
from test_interpreter import evaluate_precomputed, print_report_summary


MODEL_NAME = "Qwen/Qwen3.5-2B"  # instruct release -- NOT Qwen3.5-2B-Base


def load_qwen():
    """Loads Qwen3.5-2B in 4-bit and returns (tokenizer, model)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {MODEL_NAME} in 4-bit (cached after first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"Loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")
    return tokenizer, model


def make_qwen_llm_fn(tokenizer, model):
    """
    Returns a qwen_llm_fn(prompt: str) -> str closure — this is the exact
    seam QueryInterpreter expects (see interpreter.py: self.llm_fn(prompt)).
    """
    def qwen_llm_fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    return qwen_llm_fn


def run(limit: int | None, verbose: bool) -> None:
    tokenizer, model = load_qwen()
    qwen_llm_fn = make_qwen_llm_fn(tokenizer, model)

    interpreter = QueryInterpreter(llm_fn=qwen_llm_fn)

    bank = get_query_bank()
    if limit:
        bank = bank[:limit]
        print(f"(Running only the first {limit} queries as a smoke test)\n")

    # Per-query pass, so we can print OK/MISS + source live as it runs.
    # We keep (query, expected, actual) so the field-level report below can
    # reuse these results instead of calling Qwen on all 200 queries a
    # second time.
    source_counts = {Source.LLM: 0, Source.HYBRID: 0, Source.KEYWORD_FALLBACK: 0}
    precomputed_results = []
    for i, entry in enumerate(bank, 1):
        query = entry["query"]
        expected = entry["expected"]
        actual = interpreter.interpret(query)
        precomputed_results.append((query, expected, actual))
        source_counts[actual.source] = source_counts.get(actual.source, 0) + 1

        ok = (
            actual.task_sequence == expected.task_sequence
            and actual.domain == expected.domain
            and actual.sensor == expected.sensor
        )
        tag = "OK  " if ok else "MISS"
        print(f"[{tag}] #{i:>3} source={actual.source.value:<16} \"{query[:70]}\"")

        if not ok and verbose:
            err = interpreter.get_last_error_details()
            if err:
                print(f"       raw_llm_output: {err.get('raw_llm_output')!r}")
                print(f"       error: {err.get('error_type')}: {err.get('error_message')}")

    print("\n" + "=" * 80)
    print("Source breakdown:")
    for src, count in source_counts.items():
        print(f"  {src.value:<18}: {count} / {len(bank)}")
    print("=" * 80 + "\n")

    # Clarification-rate report: how many of these WOULD have triggered a
    # clarifying question in live use (needs_clarification=True), without
    # actually blocking this batch run on 248 terminal prompts. See
    # clarify_cli.py for the live interactive version of this flow.
    clarify_count = sum(1 for _, _, actual in precomputed_results if actual.needs_clarification)
    print(f"Would have needed clarification: {clarify_count} / {len(bank)} "
          f"({100 * clarify_count / len(bank):.1f}%)")
    print("(Batch benchmark runs never ask interactively -- see clarify_cli.py "
          "for the live version where these actually prompt the user.)\n")

    # Full field-level accuracy report, reusing results already computed
    # above -- no second pass through Qwen.
    report = evaluate_precomputed(precomputed_results, verbose=False)
    print_report_summary(report, max_failures_to_print=15)

    if source_counts[Source.KEYWORD_FALLBACK] > len(bank) * 0.3:
        print(
            "\n>30% of queries fell back to keywords — Qwen's JSON output is likely "
            "not parsing. Rerun with --verbose to see raw_llm_output per miss, then "
            "send those examples back to Person A to adjust prompt_template.py."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Print raw Qwen output on every MISS")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Only run the first N queries")
    args = parser.parse_args()

    run(limit=args.limit, verbose=args.verbose)
