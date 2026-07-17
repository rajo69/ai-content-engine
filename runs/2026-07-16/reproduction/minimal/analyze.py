"""Aggregate results.json into per-arm means, cross-seed spread, and P1-P4 verdicts.

Applies the pre-registered variance rule from ../predictions.json: no direction may be declared
from a delta smaller than the seed spread. Writes ../summary.json for the visualization stage.
"""

import json
import statistics as st

MODELS = ["gpt2", "Qwen/Qwen2.5-0.5B"]
ARMS = ["icl", "ahr", "lora"]
SEEDS = [0, 1, 2]
SEP = "-" * 72


def main():
    res = json.load(open("../results.json"))
    runs = res["runs"]

    acc, paired = {}, {}
    for m in MODELS:
        for a in ARMS:
            vals = [runs[f"{m}|{a}|{s}"]["acc"] for s in SEEDS
                    if f"{m}|{a}|{s}" in runs and "acc" in runs[f"{m}|{a}|{s}"]]
            if vals:
                acc[(m, a)] = vals
        # per-seed paired deltas: the seed picks the 10 examples, so arms are paired within it
        for a in ("ahr", "lora"):
            d = [runs[f"{m}|{a}|{s}"]["acc"] - runs[f"{m}|icl|{s}"]["acc"] for s in SEEDS
                 if f"{m}|{a}|{s}" in runs and f"{m}|icl|{s}" in runs
                 and "acc" in runs.get(f"{m}|{a}|{s}", {}) and "acc" in runs.get(f"{m}|icl|{s}", {})]
            if d:
                paired[(m, a, "icl")] = d
        d = [runs[f"{m}|ahr|{s}"]["acc"] - runs[f"{m}|lora|{s}"]["acc"] for s in SEEDS
             if "acc" in runs.get(f"{m}|ahr|{s}", {}) and "acc" in runs.get(f"{m}|lora|{s}", {})]
        if d:
            paired[(m, "ahr", "lora")] = d

    def ms(v):
        return (st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0))

    out = {"per_arm": {}, "paired_deltas": {}, "verdicts": {}}

    print(SEP)
    print(f"{'model':22s} {'arm':5s} {'mean':>7s} {'std':>6s}   per-seed")
    print(SEP)
    for m in MODELS:
        for a in ARMS:
            if (m, a) not in acc:
                continue
            mu, sd = ms(acc[(m, a)])
            out["per_arm"][f"{m}|{a}"] = {"mean": round(mu, 2), "std": round(sd, 2),
                                          "per_seed": [round(x, 2) for x in acc[(m, a)]]}
            print(f"{m:22s} {a:5s} {mu:7.2f} {sd:6.2f}   "
                  + " ".join(f"{x:.2f}" for x in acc[(m, a)]))

    print(SEP)
    print(f"{'model':22s} {'delta':16s} {'mean':>7s} {'std':>6s}   per-seed")
    print(SEP)
    for (m, a, b), d in paired.items():
        mu, sd = ms(d)
        out["paired_deltas"][f"{m}|{a}-{b}"] = {"mean": round(mu, 2), "std": round(sd, 2),
                                                "per_seed": [round(x, 2) for x in d]}
        print(f"{m:22s} {a+' - '+b:16s} {mu:7.2f} {sd:6.2f}   "
              + " ".join(f"{x:+.2f}" for x in d))

    def dmean(m, a, b):
        v = paired.get((m, a, b))
        return ms(v) if v else (None, None)

    g, q = MODELS
    verdicts = {}

    # Scoring policy, fixed here AFTER seeing results but WITHOUT touching predictions.json.
    # Two issues surfaced when scoring; both are recorded in LAB-NOTES + REPORT rather than
    # silently resolved:
    #
    #  (1) predictions.json P3 has overlapping clauses: partially_supported_if ("ordering holds,
    #      separation < 3.0") and not_supported_if ("deltas within 3.0 points") BOTH fire at
    #      separation = 1.11. Falsification takes precedence — a pre-registered falsification
    #      criterion that fires is a falsification, otherwise the criterion is unfalsifiable.
    #      analysis/core.md 7, the document predictions.json claims to transcribe, has no
    #      partial clause and says plainly: falsified if the deltas fall within 3.0 points.
    #
    #  (2) predictions.json P1/P2 diverge from their own stated source. 7 calibrates to SST-2
    #      (predicting NO AHR-LoRA margin on the weak model, paper's SST-2 basis -0.15);
    #      predictions.json used the pooled 6-dataset basis (+6.23) and predicted a margin on
    #      both. Both documents pre-date the run, so BOTH are scored below, labelled by source.
    #      Neither is quietly preferred.

    def variance_gated(delta, sd, verdict):
        """Pre-registered variance rule: no direction from a delta smaller than seed spread."""
        if sd > 0 and abs(delta) < sd:
            return verdict + " [indistinguishable at this scale: |delta| < seed std]"
        return verdict

    # --- P1 --------------------------------------------------------------------------
    gl, gls = dmean(g, "ahr", "lora")
    ql, qls = dmean(q, "ahr", "lora")
    if gl is not None and ql is not None:
        # letter of predictions.json: AHR - LoRA >= 3.0 on BOTH models
        if gl >= 3.0 and ql >= 3.0:
            v_json = "supported"
        elif gl <= 0 or ql <= 0:
            v_json = "not supported"          # not_supported_if: AHR <= LoRA on either model
        else:
            v_json = "partially supported"
        # analysis/core.md 7, SST-2-calibrated: AHR-LoRA >= 3.0 on Qwen AND |AHR-LoRA| < 3.0 on GPT-2
        v_s7 = "supported" if (ql >= 3.0 and abs(gl) < 3.0) else "not supported"
        verdicts["P1"] = {
            "verdict_as_preregistered_in_predictions_json": v_json,
            "verdict_as_specified_in_analysis_core_md_s7": v_s7,
            "scoring_conflict": v_json != v_s7,
            "gpt2_ahr_minus_lora": round(gl, 2), "gpt2_std": round(gls, 2),
            "qwen_ahr_minus_lora": round(ql, 2), "qwen_std": round(qls, 2),
            "note": ("predictions.json predicted a >=3pt AHR margin on both models; core.md s7 "
                     "predicted NO margin on the weak model and flagged the both-models form as "
                     "falsifiable-by-construction on SST-2. s7 is the better-calibrated form."),
        }

    # --- P2 --------------------------------------------------------------------------
    glo, glos = dmean(g, "lora", "icl")
    qlo, qlos = dmean(q, "lora", "icl")
    if glo is not None and qlo is not None:
        # letter of predictions.json: LoRA < ICL on >= 1 model
        v_json = "supported" if (glo < 0 or qlo < 0) else "not supported"
        # analysis/core.md 7: LoRA < ICL on Qwen AND LoRA > ICL on GPT-2 (falsified if either reverses)
        gpt2_half = variance_gated(glo, glos, "LoRA < ICL (predicted LoRA > ICL)" if glo < 0
                                   else "LoRA > ICL as predicted")
        v_s7 = "not supported" if (qlo >= 0 or glo < 0) else "supported"
        verdicts["P2"] = {
            "verdict_as_preregistered_in_predictions_json": v_json,
            "verdict_as_specified_in_analysis_core_md_s7": v_s7,
            "scoring_conflict": v_json != v_s7,
            "gpt2_lora_minus_icl": round(glo, 2), "gpt2_std": round(glos, 2),
            "qwen_lora_minus_icl": round(qlo, 2), "qwen_std": round(qlos, 2),
            "gpt2_half": gpt2_half,
            "note": ("Qwen half holds decisively (-25.83, |delta| >> seed std). GPT-2 half: s7 "
                     "predicted LoRA > ICL; observed -2.22 with std 3.14, i.e. BELOW seed spread, "
                     "so by the variance rule no direction is declarable there — the GPT-2 half is "
                     "untestable rather than cleanly reversed."),
        }

    # --- P3: THE EXPERIMENT — the gradient ---------------------------------------------
    ga, gas = dmean(g, "ahr", "icl")
    qa, qas = dmean(q, "ahr", "icl")
    if ga is not None and qa is not None:
        sep = ga - qa
        pooled = (gas ** 2 + qas ** 2) ** 0.5
        # not_supported_if fires whenever |sep| < 3.0, and takes precedence over the
        # overlapping partially_supported_if clause. See scoring policy note above.
        if sep >= 3.0:
            v = "supported"
        elif abs(sep) < 3.0:
            v = "not supported"
        else:
            v = "not supported (ordering reversed)"
        verdicts["P3"] = {
            "verdict": v,
            "gpt2_ahr_minus_icl": round(ga, 2), "gpt2_std": round(gas, 2),
            "qwen_ahr_minus_icl": round(qa, 2), "qwen_std": round(qas, 2),
            "separation": round(sep, 2),
            "separation_threshold": 3.0,
            "pooled_seed_spread": round(pooled, 2),
            "separation_below_seed_spread": bool(abs(sep) < pooled),
            "note": ("Ordering nominally holds (-2.22 > -3.33) but separation 1.11 is under BOTH "
                     "the pre-registered 3.0 threshold AND the pooled seed spread 3.42, so the "
                     "ordering itself is not declarable. predictions.json's overlapping "
                     "partially_supported_if clause is not honoured: its not_supported_if "
                     "criterion fires, and core.md s7 (the source) has no partial clause."),
        }

    # --- P4: AHR - ICL <= +3.0 on the modern model --------------------------------------
    if qa is not None:
        icl_q = st.mean(acc[(q, "icl")])
        fit = 63.60 - 0.682 * icl_q          # paper's SST-2 fit, from analysis/core.md 7
        verdicts["P4"] = {"verdict": "supported" if qa <= 3.0 else "not supported",
                          "qwen_ahr_minus_icl": round(qa, 2), "qwen_std": round(qas, 2),
                          "qwen_icl_acc": round(icl_q, 2),
                          "paper_sst2_fit_prediction": round(fit, 2),
                          "within_5_of_fit": bool(abs(qa - fit) <= 5.0)}

    out["verdicts"] = verdicts
    print(SEP)
    print(json.dumps(verdicts, indent=2))
    json.dump(out, open("../summary.json", "w"), indent=2)
    print("\nwrote ../summary.json")


if __name__ == "__main__":
    main()
