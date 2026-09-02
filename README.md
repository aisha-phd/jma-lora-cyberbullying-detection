# JMA-LoRA: Joint Multiplicative-Additive Low-Rank Adaptation for Cyberbullying Detection

Code, notebooks and results for the paper *JMA-LoRA: Joint Multiplicative-Additive Low-Rank
Adaptation for Cyberbullying Detection*, International Journal of Neural Systems.

JMA-LoRA trains a multiplicative and an additive low-rank path together,

```
W̃ = W(I + UV) + AB
```

and recovers standard additive LoRA when `UV = 0` and multiplicative LoRA when `AB = 0`. The
repository contains every experiment reported in the paper, with outputs retained so the numbers
can be read without re-running anything.

## Datasets

| | Task | Unit | Samples | Test set |
|---|---|---|---|---|
| CB1 | 6-class | Tweet | 47,692 | 10,330 |
| CB2 | Binary | Conversation | 89,525 | 20,346 |

CB1 is the SOSNet corpus (Wang et al., 2020); CB2 derives from Ejaz et al. (2024). Processed
datasets: https://github.com/aisha-phd/Cyberbullying-Detection

## Protocol

Every notebook uses the same protocol, so results are directly comparable.

- 75/25 stratified split at seed 42, with a further 10% of train held out for validation
- Near-duplicate removal at TF-IDF cosine similarity 0.90
- Four-bit NF4 quantization with double quantization; base weights frozen
- Adapters on the query, key, value and output projections, with α = r
- One epoch, class-weighted cross-entropy
- Macro F1 primary, MCC secondary, AUC-ROC additionally on CB2
- **Adapter ranks and checkpoints are selected on the validation partition. The test set is
  evaluated once, for the selected configuration only.**

## Repository layout

```
notebooks/       one notebook per experiment, outputs retained
results/         per-run CSV files written by the notebooks
figures/         framework diagram and plots
requirements.txt pinned environment
LICENSE          MIT
```

## Notebooks

| Notebook | Experiment | Reported in |
|---|---|---|
| `01_stage2_additive_rank_sweep` | Additive LoRA, r ∈ {4, 8, 16, 32} | Sec. 4.3, Table 3 |
| `02_stage3_multiplicative_rank_sweep` | Multiplicative LoRA, same sweep, plus rank-preservation diagnostics | Sec. 4.4, 4.9, Tables 3 and 8 |
| `03_stage4_jma_lora_rank_sweep` | JMA-LoRA, same sweep and diagnostics | Sec. 4.5, 4.9, Tables 3, 4 and 8 |
| `04a_stage2_multiseed` | Additive at r = 32, seeds 42/123/456 | Sec. 4.7, Table 6 |
| `04b_stage3_multiseed` | Multiplicative at r = 32, same seeds | Sec. 4.7, Table 6 |
| `04c_stage4_multiseed` | JMA-LoRA at r = 8, same seeds | Sec. 4.7, Table 6 |
| `05_peft_baseline_dora` | DoRA at r = 32 | Sec. 4.8, Table 7 |
| `05b_peft_baseline_adalora` | AdaLoRA, init_r 48 pruned to 32 | Sec. 4.8, Table 7 |
| `06_encoder_baseline_deberta` | DeBERTa-v3-large, full fine-tuning | Sec. 4.8, Table 7 |
| `07_epoch_sensitivity` | JMA-LoRA at r = 8 for three epochs | Sec. 3.2.4, 5.3 |

`05_peft_baseline_dora` also contains an earlier AdaLoRA run in which the rank allocator was never
invoked, so it trained at a fixed rank of 48 with no budget reallocation. That run is not a valid
AdaLoRA result and is superseded by `05b_peft_baseline_adalora`, which is the one reported.

## Main results

CB1, Gemma-2-9B-it, test Macro F1.

| Method | Trainable | Macro F1 | MCC | Seeds |
|---|---|---|---|---|
| DeBERTa-v3-large, zero-shot | 0 | 0.3129 | — | 1 |
| DeBERTa-v3-large, full fine-tuning | 435,067,910 | 0.9001 | 0.8932 | 1 |
| AdaLoRA, 48 → 32 | 53,703,552 | 0.8772 | 0.8686 | 1 |
| Multiplicative LoRA, r = 32 | 39,911,424 | 0.8943 | 0.8878 | 3 |
| DoRA, r = 32 | 36,298,752 | 0.9014 | 0.8942 | 1 |
| Additive LoRA, r = 32 | 35,804,160 | 0.9007 | 0.8938 | 3 |
| **JMA-LoRA, r = 8** | **18,945,024** | **0.9034** | **0.8970** | 3 |

Rows with three seeds report the mean over training seeds 42, 123 and 456. Single-seed rows are one
run at seed 42. Changing the training seed moves Macro F1 by 0.0015 to 0.0025, so differences below
roughly 0.002 should not be read as meaningful. The encoder was trained for three epochs with the
best epoch selected on validation, a more generous budget than the single epoch used for every
adapted model.

At matched trainable-parameter budgets on CB1, JMA-LoRA leads additive LoRA by +0.0082 at
approximately 9M parameters and +0.0043 at approximately 18M, and trails by 0.0051 at approximately
36M, where the additive path alone has sufficient capacity.

## Rank preservation

Proposition 2 gives ‖UV‖₂ < 1 as a *sufficient* condition for I + UV to be invertible, and hence
for rank to be preserved. Measured across the 168 adapted projections after training:

| r | max ‖UV‖₂ | layers with ‖UV‖₂ < 1 | min σ(I+UV) | max κ | effective rank |
|---|---|---|---|---|---|
| 4 | 2.22 | 31.0% | 0.368 | 7.01 | 2815.98 |
| 8 | 3.36 | 3.6% | 0.296 | 12.30 | 2815.98 |
| 16 | 6.00 | 1.2% | 0.169 | 32.51 | 2815.98 |
| 32 | 10.64 | 0.0% | 0.092 | 94.22 | 2815.55 |

The condition fails in most layers, yet effective rank is unchanged, because I + UV remains
invertible. Rank preservation rests on invertibility rather than on the Neumann bound.

## Environment

PyTorch 2.11, Transformers 4.51.3, PEFT 0.20, bitsandbytes 0.50.1, datasets 3.6, Python 3.13, on a
single A100. Each notebook pins its own environment in the first cell; run that cell, restart the
runtime, then run the remaining cells in order.

Approximate GPU time per notebook: 3 to 4 hours for a rank sweep, 7 hours for a multi-seed run,
3 hours for a PEFT baseline, 1 hour for the encoder, 7 hours for the epoch-sensitivity run.

## Citation

```bibtex
@article{saeid2026jmalora,
  author  = {Saeid, Aisha and Neri, Ferrante and Kanojia, Diptesh},
  title   = {{JMA-LoRA}: Joint Multiplicative-Additive Low-Rank Adaptation
             for Cyberbullying Detection},
  journal = {International Journal of Neural Systems},
  year    = {2026}
}
```

## Related work by the authors

- A. Saeid, D. Kanojia and F. Neri, *Decoding cyberbullying on social media: a machine learning
  exploration*, IEEE Conference on Artificial Intelligence, 2024.
- A. Saeid, A. Sabu, G. A. Koushik, F. Neri and D. Kanojia, *Cyberbullying detection via
  aggression-enhanced prompting*, RANLP 2025.
