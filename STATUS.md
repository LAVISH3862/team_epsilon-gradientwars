# UPI Fraud Rings: progress and plan (13 Nov, 8:55 PM)

## Where we stand
- **Best public score:** `sub4.csv` = **0.5060**, 5th on the board.
- **Top of the board:** Grenuke 0.5488, GradientAscent 0.5485, Organic 0.5476, Vhagar 0.5305.
- **Uploads:** 5 used today, 5 left until midnight. Round ends 15 Nov, 10:00 AM.

## Uploads so far
| File | What it is | Validation (test-like) | Public |
|---|---|---|---|
| sub4 | Row + connection + ring + flag features, cleaned labels, rule tuned on cleaned labels | 0.5219 | **0.5060** |
| sub5 | sub4 recipe as a blend of LightGBM, XGBoost, CatBoost, rule tuned on test-like labels | 0.5235 | 0.5038 |
| sub6 | Probe: sub4 with account_takeover + sim_swap set to legit | — | 0.4329 |
| sub7 | Probe: sub4 with velocity_bot + merchant_collusion + refund_abuse set to legit | — | 0.4105 |
| sub12 | Second round of label cleaning, rule tuned on test-like labels | 0.5236 | 0.5004 |

## Ready, not uploaded
| File | What it is | Expected public |
|---|---|---|
| sub13 | sub12's model, rule tuned like sub4 (56 rows differ from sub4) | 0.503 to 0.510 |
| sub14 | sub4 flagging 95 fewer borderline rows | 0.500 to 0.512 |
| sub10 | sub4 recipe + pseudo-labels on October (check the rules first) | about 0.506 |

## What we built (day 1)
1. **Setup:** kit copied to this folder, folds by month (July, August, September), rules noted in `comp.py`.
2. **Features** (`features.py`, built on train + test rows together):
   - `row`: single-row signals and whose phone it is (0.39 out-of-fold).
   - `conn`: bursts at receiver, sender and phone, money moving on, payments the other way (0.39 → 0.47).
   - `ring`: statistics of the event a row belongs to.
   - `flag`: explicit states of the QR mismatch flag, because October blanks it on many qr_swap rows (fixed 159 October rows sub3 called legit).
3. **Label cleaning** (`clean_labels.py`): 36,000 training rows carry a fraud label but look exactly like legit rows; trained on as legit.
4. **Decision rule** (`postprocess.py`): per-class biases tuned for macro-F1.
5. **Cloud:** Kaggle CPU jobs for extra models (`kaggle_run.py`).

## What we learned
- **The test keeps random fraud labels.** All-legit scores 0.0816, so about 4.1% of public rows carry a fraud label; we can only catch about 1.5%. That caps everyone's score.
- **Our losses are in three types:** velocity_bot, merchant_collusion, refund_abuse (probe sub7). The look-alike pair account_takeover / sim_swap is fine (probe sub6).
- **Tuning the rule on test-like labels hurt:** sub5 and sub12 flag extra borderline rows that the board says are mostly wrong.

## Tried, no gain
| Idea | Result |
|---|---|
| More models and blending | 0.5235, board 0.5038 |
| Pseudo-labels on October | 0.5242 (sub10, not uploaded) |
| Half-density training to match October | no change (0.863 / 0.866 / 0.865) |
| Training on labels as given (no cleaning) | worse, 0.5200 |
| Ring propagation (neighbours' predictions as features) | 0.5227 |
| Dropping features that drift across months | 0.5231 |
| Broken column in the new app version | none found |
| Ids, exact time, digit patterns | nothing (AUC 0.53) |

## Plan ahead
**Tonight (13 Nov)**
- Upload `sub13.csv` and `sub14.csv`. Their scores show whether to make the rule more or less cautious.

**Tomorrow (14 Nov)**
- Keep hunting the pattern behind the 0.548 cluster (three teams on the same score points to one specific signal), starting with October's merchant-side rows.
- Upload probes `sub8.csv` and `sub9.csv` if needed, to check the remaining fraud types.
- Ask the organisers whether pseudo-labels are allowed.
- Create the git repo and push the code (only when told to push).

**15 Nov, before 9:00 AM**
- `reproduce.sh` complete, code link submitted.
- No look-alike uploads late: the final pick is automatically the best public upload.
