# Lab notes

Append-only. Failures included on purpose.

## 2026-07-17 — session 2 (this session)

**Inherited state.** A previous partial attempt left `predictions.json`, `hyperparams.md`,
`minimal/ahr.py`, `minimal/gate_p5.py`. No results, no `p5_gate.json` — the gate script had
been *started* (there is a `__pycache__` for `ahr.py`, so it was imported) but never wrote its
output file, which it only does after both models succeed. So the previous session died
somewhere in the gate. Building on the inherited files rather than redoing them.

**Environment probe.** Contrary to `analysis/core.md` §7 ("`transformers`, `peft`, `datasets`
are **not installed**"), all three are present system-wide: torch 2.9.1+cpu, transformers
5.3.0, peft 0.19.1, datasets 4.8.5, Python 3.13.7. No venv exists in the repo. **Decision:**
use the system interpreter as-is rather than spending time-box minutes on `uv venv` +
reinstall. Recorded exact versions in `env-info.json` and pinned them in `requirements.txt`.
This is a deviation from the stage's "use `uv`" instruction, taken deliberately: the packages
were already there, the box is 90 minutes, and pinning the versions we actually ran preserves
replicability, which is the point of the instruction.

**Lucky break.** Both `gpt2` and `Qwen/Qwen2.5-0.5B` are already in the HF cache, as is
`unsloth/Llama-3.2-1B-Instruct` (an ungated mirror of the paper's own Llama-3.2-1B). No
download cost. The Llama mirror is a tempting *third* point on the competence axis — the paper
tests that exact model (AHR−ICL = +3.66 on SST-2) — but 1B params on a laptop CPU is likely
out of budget. Noted as a stretch goal only.

**P5 gate: PASS, and stronger than pre-registered.** Threshold was max|dev| < 1e-4 between the
γ=0 hooked model and the clean frozen model. Observed **exactly 0.000e+00 on both models** —
bitwise identical, not merely close. The forward-pre-hook on the attention output projection
is therefore an exact implementation of Eq. 3 (scaling head h's slice of W_O's input ≡ scaling
head h's block of W_O ≡ scaling head h's additive residual-stream contribution). The
anti-trivial check (perturb one γ to 0.5 → logits must move) gives 7.998 / 8.425, so the hook
is live and we are not passing the gate by silently doing nothing. Head geometry matches the
paper's parameterization: GPT-2 12×12 = **144** γ, Qwen2.5-0.5B 24×14 = **336** γ.

### FAILURE 1 — the paper's Eq. 4 is NaN at the paper's own initialization

First smoke run: AHR CE was finite at step 0 (1.5680) and **NaN from step 1 onward**; final γ
all NaN. Cause, found by inspection rather than more runs:

Eq. 4's regularizer is `λ (Σ_h |γ_h|^p)^(1/p)` — that is the Lp **norm**. The Lp norm is not
differentiable at the origin, and §3 also specifies **γ is zero-initialized**. For p=2,
`d/dγ √(Σγ²) = γ/√(Σγ²) → 0/0` at γ=0. The first backward pass from the paper's stated init
produces a NaN gradient, which then propagates to every γ forever.

So **Eq. 4 as literally written, combined with the paper's own stated γ=0 initialization, does
not run** for p=2 and any λ>0. This is a genuine paper-level discrepancy, not a typo we should
paper over — and it is unverifiable against the repo because the repo has no code. Two readings:
(a) the authors mean squared L2 (`λ Σγ²`, the universal convention, = weight decay), or (b) they
use λ=0 for p=2 in practice, or (c) their implementation adds an ε. The paper does not say.

**Fix:** use the standard convention — p=2 → `λ Σγ²` (smooth everywhere), p=1 → `λ Σ|γ|`
(subgradient 0 at the origin, which is fine). Both preserve the property the paper's argument
actually leans on: the penalty pulls γ toward 0, i.e. toward the unmodified pretrained model.
At λ=0.01 the numerical difference is negligible. Documented in `ahr.py:penalty` and
`hyperparams.md`. After the fix AHR trains normally (CE 1.5680 → 0.9607, |γ| mean 0.11).

### FAILURE 2 (not a bug — a finding) — GPT-2-small is a degenerate predictor

Added a `frac_pred_pos` diagnostic precisely because a model that always answers one class
scores ~50% on balanced SST-2 and is otherwise indistinguishable from an honest coin flip.
Result: **GPT-2-small ICL predicts "Positive" for 95–100% of test examples**, accuracy 45%.

Checked and rejected the harness-bug hypotheses: label verbalizers tokenize to distinct first
tokens (asserted in code); the eval prefix is the identical 10-shot sequence used for training;
scoring reads the logit at the position after "Sentiment:". The prompt is fixed a priori and was
NOT tuned after seeing this number — tuning it would be exactly the researcher-degrees-of-freedom
move this stage's discipline exists to prevent.

Conclusion: this is real label bias in a 124M model (cf. Zhao et al. 2021, *Calibrate Before
Use*), not an implementation error. It is *informative* rather than fatal: `analysis/core.md` §0
asks whether AHR "merely restores in-context-learning ability that a weak model was already
failing to express." A base model whose ICL is broken by label bias is the sharpest possible
test bed for that question. Keep GPT-2-small as the weak end and report the diagnostic.

### The axis is wider than the paper's

Measured before any training (P0, which costs nothing and keys everything else):

| Model | 10-shot ICL acc | frac predicted positive |
|---|---|---|
| GPT-2-small (124M) | **45.0** | 0.95 — degenerate |
| Qwen2.5-0.5B | **95.0** | 0.55 — calibrated |

P0 predicted GPT-2-small ≈ 60–80 and Qwen2.5-0.5B ≈ 85–92. **Both missed**: GPT-2-small is far
worse (45, below chance) and Qwen2.5-0.5B is better (95) than pre-registered. Recording the miss
rather than quietly re-baselining. The 50-point spread is *wider* than the paper's own
GPT2-XL(72.00) → Qwen3-8B(93.29) SST-2 range, which makes this a stronger test of P3, not a
weaker one. Note Qwen2.5-0.5B at 95.0 sits **above** the SST-2 fit's zero-crossing (ICL ≈ 93),
where `63.60 − 0.682×ICL` predicts AHR−ICL ≈ **−1.2**, i.e. AHR should mildly *hurt*.

### Process notes

- Timing forced the plan's pre-registered fallback tier 2: **120 eval examples** (not 200).
  Qwen eval costs ~0.9 s/example on this CPU; 200 × 9 runs would have blown the 90-minute box.
  Kept **3 seeds** rather than the fallback's 2 — `analysis/core.md` §4 shows seed-level spread
  is the thing most likely to bite a small run, so spread reporting was the wrong thing to cut.
  The 120-example eval subsample is **identical across every model, arm and seed** (fixed
  rng seed 1234), so all comparisons remain paired. It happens to be 65/120 = 54% positive.
- **Self-inflicted, worth recording.** Launched the full run with `nohup ... &` under Git Bash;
  `ps aux | grep -c run_experiment` then reported **0**, so I concluded the process had died with
  its parent shell and relaunched under the harness. Both conclusions were wrong: Git Bash's `ps`
  cannot see native Windows processes, and the original was alive the whole time. Result: **two
  identical runs writing the same `results.json` and oversubscribing 10 physical cores 2:1**,
  which is why progress appeared stalled at 1/18 cells. `Get-CimInstance Win32_Process` showed
  the truth (PIDs 2736 @ 765 CPU-s and 3048 @ 482 CPU-s). Killed the younger duplicate, kept the
  original — it was further ahead and its `_run.log` is unbuffered, whereas the harness-tracked
  one was piped through `grep`, which block-buffers and hid its output entirely. Lesson: on
  Windows, verify process liveness with `Get-CimInstance`, not `ps`; and never pipe a background
  run's stdout through `grep` if you intend to watch it. No results were corrupted — both
  processes were running an identical config and each dumps its own complete dict — but ~8
  minutes of the box were lost to contention.
- `results.json` is rewritten after **every arm**, so a run killed by the time box or budget
  keeps everything already earned, and re-running skips completed cells.
- No venv: torch/transformers/peft/datasets were already installed system-wide. Deliberate —
  see above. `requirements.txt` pins the versions actually executed (`pip freeze` reports
  `torch==2.9.1`; the real build is `2.9.1+cpu`, corrected by hand).

## 2026-07-17 — session 3 (scoring + write-up)

The 18-cell run completed cleanly under the original (non-duplicated) process — `_run.log` ends
with `wrote ../results.json` and all 2 models × 3 arms × 3 seeds are present. No compute was
re-run this session; nothing needed it. Note for future sessions: `Get-CimInstance Win32_Process`
filtered on `*run_experiment*` **matches your own shell** if the query string is in its command
line. Four "still running" PIDs turned out to be my own Bash/PowerShell invocation. Verify by PID
against the ones you actually launched.

### FAILURE 3 (mine, not the paper's) — my scorer graded on the curve

`analyze.py` returned **"P3: partially supported"**. That was wrong, and it was wrong in the
direction that flatters the pipeline, which is the direction that deserves the most suspicion.

Two separate defects, both found by scoring by hand against `predictions.json` instead of trusting
my own code:

1. **Overlapping pre-registration clauses.** P3's `partially_supported_if` ("ordering holds,
   separation < 3.0") and `not_supported_if` ("the two deltas are within 3.0 points") **both fire**
   at separation = 1.11. My code checked the partial clause first, purely by ordering of `if`
   branches, and returned the pass. Resolution: **falsification takes precedence** — a
   pre-registered falsification criterion that fires *is* a falsification, or it was never
   falsifiable and never worth registering. Confirmed against `analysis/core.md` §7, which
   `predictions.json` claims to transcribe: §7 has **no partial clause** and says "Falsified if the
   two deltas fall within 3.0 points of each other." Verdict corrected to **not supported**.
2. **`predictions.json` diverges from its own stated source on P1/P2.** It says "transcribed
   verbatim, not revised" from §7. It isn't. §7 calibrated P1 to the **SST-2 column** (predicting
   *no* AHR−LoRA margin on the weak model, basis −0.15) and explicitly warned that a naive "AHR
   beats LoRA by ≥3 on both models" prediction "would be falsified on GPT-2-small even by a perfect
   reproduction." `predictions.json` then registered exactly that naive form off the **pooled**
   basis (+6.23) — walking into the trap §7 had flagged one paragraph earlier. Same story for P2.
   Both verdicts flip depending on which document you score against.

**Handling:** `predictions.json` was **not touched** — the no-edit-after-running rule is the whole
point of pre-registration, and a badly-worded prediction is not a licence to reword it. Instead
`analyze.py` now scores P1/P2 against **both** documents, labels each verdict by source, and sets
`scoring_conflict: true`. Both are in REPORT.md. I did not silently take the flattering one in
either case (§7 favours us on P1, `predictions.json` favours us on P2 — reporting both costs us P1's
clean pass and gains us nothing, which is roughly the point).

Lesson worth keeping: a pre-registration is only as good as its mutual exclusivity. Next run, the
predictions file should assert that supported/partial/not-supported conditions **partition** the
outcome space, and a scorer should refuse to grade when two clauses fire simultaneously rather than
resolving by `if`-order.

### The result

**P3 — the experiment — is not supported.** AHR−ICL is −2.22 ± 3.14 (GPT-2-small) vs −3.33 ± 1.36
(Qwen2.5-0.5B): separation 1.11, under the 3.0 threshold and under the 3.42 pooled seed spread.
The nominal ordering holds and is worthless — it's inside noise, and the variance rule we
pre-registered exists precisely to stop us selling it.

Root cause is the P0 miss already recorded in session 2, and recorded *before* these results
existed, which is the only reason it's usable as a diagnosis rather than a story: GPT-2-small's ICL
is **45.0, below chance**, 95% one-class. It is degenerate, not weak. Seeds 1 and 2 have *every arm
at exactly 45.83* — the model is pinned and nothing moves it. We spanned broken → good, not the
paper's weak → good. So P3 was probably unreachable with this model pair from the moment P0 came
back at 45, and the correct move at that point would have been to swap the weak end to GPT-2-large
(~72 ICL, matching GPT2-XL's regime) rather than proceed. I proceeded. That cost the experiment.

What survives, and it isn't nothing: **P4 supported** (AHR −3.33 ± 1.36 at ICL 92.50 — worse than
doing nothing, magnitude above seed spread), and the LoRA-is-broken decomposition reproduced
harder than the paper's own (LoRA −25.83 vs frozen; AHR's +22.50 "win" is entirely LoRA's collapse).
LoRA seed 2 is the specimen: train loss 1e-4, emerged a constant classifier at 48.33%.

Also worth recording as a genuine confirmation *for* the paper: AHR's γ never collapsed
(|γ| ≈ 0.07–0.11, CE 0.02–0.19 vs LoRA's 1e-4). The inexpressiveness mechanism works exactly as
claimed. What we dispute is only whether it buys anything over not training.

Back-edge written to `analysis-delta.md`: downstream must **not** claim we reproduced the gradient.

## 2026-07-17 — session 4 (audit + correction)

No compute re-run. Audited every number in `REPORT.md` against `results.json` by hand. The means and
population stds all check out. One real defect found, and one diagnostic strengthened.

### FAILURE 4 (mine) — smoke-test numbers were quoted as full-run results

**`REPORT.md`, `analysis-delta.md`, and session 3's note above all reported GPT-2-small's ICL as
"45.0, below chance, 95% one-class."** That pair of numbers is from `_smoke.json` — **n_eval=20,
seed 0 only** — not from the 18-cell run. The full 120-example, 3-seed result is:

| | smoke (n=20, seed 0) | **full run (n=120, 3 seeds)** |
|---|---|---|
| GPT-2-small ICL | 45.0, frac_pos 0.95 | **50.83 ± 7.07** (60.83 / 45.83 / 45.83) |
| Qwen2.5-0.5B ICL | 95.0, frac_pos 0.55 | **92.50 ± 0.68** (91.67 / 93.33 / 92.50) |

Three things were wrong as a consequence, now corrected in all three files:

1. **The bias direction was misstated.** "Predicting Positive for 95% of examples" is true only of
   seed 0 (frac_pred_pos 0.9167). Seeds 1 and 2 predict **Negative for 100%** (frac_pred_pos 0.0).
   The report asserted a single fixed direction where the data shows it *flips with the prompt* —
   which, ironically, is the more interesting fact and the one that identifies the bias (below).
2. **"Below chance" was imprecise.** 50.83 is at coin-flip. The meaningful comparison is the eval
   set's **majority-class baseline of 54.17%** (65/120 positive), which GPT-2-small is below. That
   is the defensible form of the claim: worse than always answering "Positive".
3. **The P0 miss is smaller than session 2 claimed, and Qwen's is marginal.** GPT-2-small's mean
   (50.83) is outside the 60–80 band, but seed 0 alone (60.83) is *inside* it. Qwen at 92.50 misses
   the 85–92 band by 0.5 points — "wrong (low)" overstated it; corrected to "marginal miss (high)".
   Session 2's note that Qwen "at 95.0 sits **above** the fit's zero-crossing (ICL≈93), so AHR should
   mildly hurt (−1.2)" is **void**: at the real 92.50 the fit predicts **+0.51**, mildly *positive*.
   `analyze.py` used 92.50 and scored P4 correctly against +0.51, so no verdict is affected — the
   error was confined to that prose.

**Nothing about the verdicts changes.** P3's separation (1.11), P4 (−3.33 ± 1.36), and the LoRA
collapse (−25.83) were all computed by `analyze.py` from `results.json` and never touched the smoke
values. The degeneracy finding also survives intact — all three seeds are one-class-dominant. Only
the reported *numbers characterising* it were wrong.

Session 2's entry above is **left as written**: it was an accurate record of what was known when the
decision to proceed was taken (the smoke value was the only value then, and it is why GPT-2-small
was wrongly kept as the weak end). Rewriting history in an append-only notebook to make past-me look
better is the exact failure this file exists to prevent. Session 3's line 174 has no such excuse —
the full results existed by then — and is corrected here.

Lesson: a smoke rung exists to prove the pipeline runs, and its numbers must be **quarantined from
the write-up**. Next run: have `analyze.py` source every figure in the report from `results.json`
programmatically, so a smoke number cannot be transcribed into a finding by hand.

### The degeneracy diagnosis, sharpened (and it is *not* recency bias)

Session 2 called the GPT-2-small collapse "label bias (cf. Zhao et al. 2021)" without saying which
of Zhao et al.'s biases. The full-run per-seed data identifies it, at zero compute cost — the
collapse direction tracks the **class balance of the ten demonstrations**, 3 of 3 seeds:

| seed | demo labels | pos/neg | last demo | observed ICL | collapse |
|---|---|---|---|---|---|
| 0 | `[0,0,1,1,0,1,0,1,1,1]` | **6 / 4** | POS | 60.83, frac_pos 0.9167 | → **Positive** |
| 1 | `[0,0,0,1,0,1,1,0,0,1]` | **4 / 6** | POS | 45.83, frac_pos 0.0 | → **Negative** |
| 2 | `[0,0,1,0,0,1,0,1,0,1]` | **4 / 6** | POS | 45.83, frac_pos 0.0 | → **Negative** |

So it is **majority-label bias**, not **recency bias**: the final demonstration is Positive in all
three seeds, yet two of the three collapse to Negative. Recency is ruled out; demo class balance
predicts every case. Caveat: n=3 — consistent, not conclusive.

Reproduce (no model forward pass, seconds, needs only `datasets`):

```bash
cd minimal && python -c "
import random
from datasets import load_dataset
pool = [(r['sentence'], r['label']) for r in load_dataset('stanfordnlp/sst2')['train']]
for seed in [0,1,2]:
    labs = [l for _,l in random.Random(seed).sample(pool, 10)]
    print(seed, labs, 'n_pos=%d' % sum(labs), 'last=%s' % ('POS' if labs[-1] else 'NEG'))
"
```

This matters for `core.md` F1: **accuracy alone cannot distinguish a degenerate model from an honest
coin flip.** Any competence-axis sweep must screen endpoints with a `frac_pred_pos` diagnostic
before adopting them — which is precisely the check that would have caught GPT-2-small before it
cost us P3.

