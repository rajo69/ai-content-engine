"""P5 correctness gate: gamma = 0 must reproduce base-model logits.

Eq. 3 collapses to Eq. 1 when gamma = 0, so an AHR-hooked model at initialization must be
*exactly* the frozen model. This is the one place a silent bug would invalidate every
downstream number. Pre-registered threshold: max abs deviation < 1e-4.

Also checks that a non-zero gamma actually changes the logits (i.e. the hook is live and
we are not trivially passing the gate by doing nothing).
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ahr import AHR

MODELS = ["gpt2", "Qwen/Qwen2.5-0.5B"]
TEXT = "Review: a stirring, funny and finally transporting film.\nSentiment:"


def main():
    torch.set_num_threads(8)
    results = {}
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        model.eval()
        ids = tok(TEXT, return_tensors="pt")

        with torch.no_grad():
            base = model(**ids).logits

        ahr = AHR(model)
        with torch.no_grad():
            zero = model(**ids).logits
        dev_zero = (base - zero).abs().max().item()

        # sanity: hook must be live
        with torch.no_grad():
            ahr.gamma[0, 0] = 0.5
            perturbed = model(**ids).logits
            ahr.gamma[0, 0] = 0.0
        dev_perturbed = (base - perturbed).abs().max().item()

        n_layers, n_heads, head_dim = ahr.n_layers, ahr.n_heads, ahr.head_dim
        ahr.remove()

        passed = dev_zero < 1e-4 and dev_perturbed > 1e-6
        results[name] = {
            "n_layers": n_layers,
            "n_heads_per_layer": n_heads,
            "head_dim": head_dim,
            "n_trainable_gamma": n_layers * n_heads,
            "max_abs_dev_gamma_zero": dev_zero,
            "max_abs_dev_gamma_perturbed": dev_perturbed,
            "passed": bool(passed),
        }
        print(f"{name}: layers={n_layers} heads/layer={n_heads} head_dim={head_dim} "
              f"gamma_params={n_layers*n_heads}")
        print(f"  gamma=0   max|dev| = {dev_zero:.3e}  (gate: < 1e-4)")
        print(f"  gamma!=0  max|dev| = {dev_perturbed:.3e}  (sanity: > 1e-6)")
        print(f"  => {'PASS' if passed else 'FAIL'}")
        del model

    all_passed = all(r["passed"] for r in results.values())
    with open("../p5_gate.json", "w") as f:
        json.dump({"all_passed": all_passed, "per_model": results}, f, indent=2)
    print("\nP5 GATE:", "PASS" if all_passed else "FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
