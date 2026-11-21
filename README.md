# Key release under the sustain pedal in MAESTRO: analysis code and results

Analyses for the manuscript *Waiting for Each Other: A Two-Way Hand–Foot Relay at Sustain-Pedal Changes in 201 Competition Pianists*
(originally planned as *Control Follows Audibility: Key Release Under the Sustain Pedal in 1,276 Expert Piano Performances*).
They ask whether pianists control key releases less when the sustain pedal makes them inaudible (H2, H3), and how the hand and foot
coordinate at pedal changes (H1).

Data: MAESTRO v3.0.0 MIDI (CC BY-NC-SA 4.0, not included) and the cleaned metadata of the una corda study
(`una_corda_study/data/metadata_clean.csv`, `features.csv`).

## Run order

```bash
python -I code/19_build_note_cache.py <maestro-v3.0.0 dir>        # notes, key releases, pedal controllers -> results/derived/
python code/20_synthetic_validation.py                            # freeze plan (SHA-256), synthetic recovery, power
python code/22_extract_transitions.py                             # melodic transitions, chords, data quality (+ robustness variants)
python code/23_handover.py                                        # H1 (+ una corda control, E3)
python code/26_models.py part=study1                              # H2, study 1 (within performance)
python code/27_specificity.py part=study1                         # H3a, H3b, E1 (within performance)
python code/24_alignment_paths.py                                 # DTW paths for all pieces with >= 2 performances
python code/25_matched_transitions.py                             # same score position across performances
python code/26_models.py part=study2                              # H2, study 2 (two-way fixed effects)
python code/27_specificity.py part=study2                         # H3a, H3b at the same score position
python code/28_individuality.py                                   # E2, E4
python code/29_robustness.py                                      # robustness + Holm-corrected confirmatory table
python code/30_figures.py                                         # Figures 1-4 (greyscale, PNG 300 dpi + PDF)
python code/31_verify_numbers.py                                  # independent recomputation of headline numbers
# extensions (designed after a pilot on the same data; exploratory): how the handover is organised
python code/32_handover_events.py                                 # event / held-note tables, same-position matching
python code/33_handover_mechanisms.py                             # D1 tracking, D2 stroke coupling, D3 what gets cut, D4 two-process model
python code/34_mechanism_figures.py                               # Figures 5-7
python code/35_verify_mechanisms.py                               # independent recomputation for D1-D3
# second round (exploratory): two-way relay, anticipatory coupling, reference point, release prediction
python code/36_relay_and_control.py                               # parts relay, ctrl, ref, pred
python code/37_round2_figures.py                                  # Figures 8-9
python code/38_verify_round2.py                                   # independent checks for step 36
# checks run before writing the paper, and the paper's figures
python code/39_supplementary_checks.py                            # edition-by-edition replication; local-tempo control of the shared-plan slopes
python code/40_paper_figures.py                                   # figures/paper/Fig1-Fig7 for the manuscript
python code/41_review_checks.py                                   # review: tempo-matched / local surrogates, windows, anchor check, reliability
python code/42_review2_checks.py                                  # second review: placebo-corrected foot indices, precision weighting, early-release latency
```

`release_common.py` holds every parameter (`PLAN`) and shared estimator. Its SHA-256 hash is recorded in
`results/analysis_plan_frozen.json`, frozen before steps 22-29 were run. The study was **not** publicly preregistered;
`results/deviations_from_plan.md` lists every departure from the written plan. `results/derived/` (about 1.9 GB of
intermediate pickles) is not distributed and is rebuilt by steps 19-25.

## Main result files

| File | Content |
| --- | --- |
| `results/confirmatory_tests.csv` | The eight planned tests with Holm-adjusted p values |
| `results/h1_summary.json`, `h1_performance_indices.csv`, `h1_density_curves.csv` | Handover (H1), cut notes (E3), una corda control |
| `results/study1_summary.json`, `study1_performance_effects.csv` | Within-performance release effects (study 1) |
| `results/study2_summary.json`, `study2_design_ladder.json` | Same-score-position release effects (study 2) |
| `results/specificity_summary.json` | H3a, H3b, E1 |
| `results/individuality_summary.json` | E2, E4 |
| `results/robustness_table.csv` | All robustness variants and leave-one-edition-out runs |
| `results/synthetic_validation.json`, `power.json` | Estimator checks on synthetic data |
| `results/verification.json` | Independent recomputation |
| `results/mechanisms_summary.json`, `d1_tracking_slopes.csv`, `d2_stroke_slopes.csv`, `d4_mixture_parameters.csv` | Handover mechanisms (directions 1-4) |
| `results/verification_mechanisms.json` | Independent recomputation for the mechanisms |
| `results/round2_summary.json`, `relay_foot_indices.csv`, `reference_point.csv`, `depth_at_release.csv`, `release_prediction.csv` | Second round (relay, anticipatory coupling, reference point, prediction) |
| `results/verification_round2.json` | Independent checks for the second round |
| `results/supplementary_checks.json`, `edition_replication.csv` | Edition replication and local-tempo control (step 39) |
| `results/review_checks.json`, `review_hand_indices.csv`, `review_foot_indices.csv` | Checks requested in the internal review (step 41) |
| `results/review2_checks.json` | Checks requested in the second internal review (step 42) |

## Where each result of the paper comes from

Paper: *Waiting for Each Other: A Two-Way Hand–Foot Relay at Sustain-Pedal Changes in 201 Competition Pianists*
(Psychological Research submission version).

| Paper element | Script(s) | Main result files |
| --- | --- | --- |
| Corpus, events, Table 1 | `19`, `22`, `23`, `32`, `36` | `data_quality.csv`, `h1_summary.json`, `round2_summary.json` |
| Sample size and a priori power (Method) | `20` | `power.json`, `synthetic_validation.json` |
| Hand waits for foot (H1), Table 2, Fig. 2B, Table 3 (hand) | `23`, `29` | `h1_summary*.json`, `h1_density_curves.csv`, `robustness_table.csv`, `confirmatory_tests.csv` |
| Foot waits for hand, Fig. 2A, Table 3 (foot) | `36` (part relay), `40` | `round2_summary.json`, `relay_foot_indices.csv` |
| Stricter surrogates, windows, placebo correction, precision (Table 3, Online Resource Table S3) | `41`, `42` | `review_checks.json`, `review2_checks.json` |
| Ten editions, individual differences, reliability (Fig. 3, Table S1) | `39`, `41`, `28`, `40` | `edition_replication.csv`, `review_checks.json`, `individuality_summary.json` |
| Tracking and stroke coupling (Fig. 4, Table 4) | `32`, `33`, `34` | `mechanisms_summary.json`, `d1_*.csv`, `d2_*.csv` |
| Reference point and two-process model (Fig. 5A, Table 4) | `36` (part ref), `33` (D4), `40` | `reference_point.csv`, `d4_mixture_parameters.csv` |
| Central coupling: habit and score-position slopes, tempo control, anchor check, latency (Fig. 5B, Table 4, Table S4) | `36` (part ctrl), `39`, `41`, `42`, `40` | `round2_summary.json`, `supplementary_checks.json`, `review_checks.json`, `review2_checks.json` |
| Which notes are released early (Fig. 6) | `33` (D3), `34` | `mechanisms_summary.json` |
| Audibility account H2/H3 (Table 2, Fig. 7, Appendix A) | `22`, `24`–`29`, `30` | `study1_summary*.json`, `study2_summary*.json`, `study2_design_ladder.json`, `specificity_summary.json` |
| Release prediction and pedal values across editions (Appendix B) | `36` (parts pred, ref), `38` | `release_prediction.csv`, `depth_at_release.csv`, `verification_round2.json` |
| Schematic (Fig. 1) | `40` | `paper_figure_values.json` |
| Independent recomputation (Method) | `31`, `35`, `38` | `verification*.json` |

Figures of the paper: `figures/paper/Fig1`–`Fig7` (written by `40_paper_figures.py`, which copies Fig. 4 and Fig. 6 from step 34 and Fig. 7 from step 30).
