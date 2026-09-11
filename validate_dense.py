import argparse
import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from santa_backend import CONTROLLER, register_backend


def parse_args():
    p = argparse.ArgumentParser(description="Compare native HF SDPA with the custom SANTA backend in dense mode.")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-rel-l2", type=float, default=5e-3)
    p.add_argument("--max-abs", type=float, default=0.20)
    return p.parse_args()


def encode_chat(tok, prompt):
    messages = [
        {"role": "system", "content": "You are a concise helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    enc = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    return {k: v.cuda() for k, v in enc.items()}


def run_backend(model_name, tok, impl, prompts, max_new_tokens):
    if impl == "santa_systematic":
        CONTROLLER.configure("dense")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        attn_implementation=impl,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    records = []
    with torch.inference_mode():
        for prompt in prompts:
            enc = encode_chat(tok, prompt)
            n = enc["input_ids"].shape[-1]
            logits = model(**enc, use_cache=False).logits[0, -1].float().cpu()
            next_id = int(logits.argmax().item())
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0, n:], skip_special_tokens=True).strip()
            records.append({"prompt": prompt, "logits": logits, "next_id": next_id, "text": text})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def main():
    args = parse_args()
    register_backend()
    tok = AutoTokenizer.from_pretrained(args.model, token=os.getenv("HF_TOKEN"))
    prompts = [
        "What is the capital of France? Answer with only the city name.",
        "Calculate 17 times 23. Answer with only the number.",
        "Which planet is known as the Red Planet? Answer with only the planet name.",
    ]

    print("Running native SDPA reference...", flush=True)
    native = run_backend(args.model, tok, "sdpa", prompts, args.max_new_tokens)
    print("Running custom backend in dense mode...", flush=True)
    custom = run_backend(args.model, tok, "santa_systematic", prompts, args.max_new_tokens)

    passed = True
    for i, (a, b) in enumerate(zip(native, custom), 1):
        diff = b["logits"] - a["logits"]
        max_abs = float(diff.abs().max().item())
        rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(a["logits"]).clamp_min(1e-12))
        argmax_same = a["next_id"] == b["next_id"]
        text_same = a["text"] == b["text"]
        ok = max_abs <= args.max_abs and rel_l2 <= args.max_rel_l2 and argmax_same and text_same
        passed &= ok
        print(f"\n[{i}] {a['prompt']}")
        print(f"  max_abs={max_abs:.6g} rel_l2={rel_l2:.6g} argmax_same={argmax_same} text_same={text_same}")
        print(f"  native: {a['text']!r}")
        print(f"  custom: {b['text']!r}")
        print(f"  result: {'PASS' if ok else 'FAIL'}")

    print("\nDENSE BACKEND VALIDATION:", "PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
