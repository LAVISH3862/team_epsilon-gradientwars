# UPI Fraud Rings: plan (ML Gradient Wars 2026)

Round: 13 Nov 10:00 AM → 15 Nov 10:00 AM IST. Results 15 Nov 5:00 PM.

## Rules that shape the plan
- Metric: macro-F1 over 12 labels. All-legit scores about 0.08; 0.40 is "excellent".
- 10 uploads a day, board refreshes every 15 minutes. Public = 30% of test, private = 70%.
- Final = private score of our **best public upload** (automatic), so every upload can become the final.
- Train only on the provided data; public pretrained weights are fine; no hand-labelling of test.
- Using `test.csv` to build connections between rows is allowed and expected.
- Code link (GitHub or notebook) must be submitted before 15 Nov 10:00 AM and reproduce the best upload.

## What the first look at the data found (13 Nov)
- Train: 1,564,060 rows, Jul–Sep plus 147 labelled rows from 1–10 Oct. Test: 260,582 rows, Oct plus 33 rows in Nov.
  No missing values except `merchant_category` (P2P rows) and `qr_vpa_mismatch` (non-QR rows).
- 96.3% legit. Fraud classes are 0.13% to 0.65% each, stable across the three months.
- **Columns that change in October:** `app_version` is 3.0–4.8 in train and only 5.0–5.2 in test;
  `ip_risk_score` has a different distribution in test. Both go out or get re-scaled per month.
- `sim_age_hours` and `device_age_days` jump around for the same account and phone, so they are noisy per row.
- **Labels are very noisy.** 79% of fraud-labelled rows have no other fraud row on the same sender, phone or
  person-receiver that day; busy merchants carry about 2.4% fraud labels of all 11 types. The real rings are
  clearly visible underneath: phone bursts (velocity_bot), money passed on minutes later (layering_chain,
  mule_account), runs of REFUND claims (refund_abuse), round-number sales to one merchant (merchant_collusion).
  Rows inside a ring are sometimes labelled legit or as another type.
- **Phone ownership is a signal:** an account's own phone has the same number (`AC377784` ↔ `DV0377784`). Phones
  numbered 420000+ are extra phones: 37% of velocity_bot rows and 25% of account_takeover rows, against 3% of legit.
- 91–95% of test rows involve a sender, receiver or phone already seen in train.

## Validation
- Folds by month (Jul / Aug / Sep), never random: rows of one ring must stay together and test is a later month.
  Headline number: train Jul–Aug → validate Sep. Leave-one-month-out gives out-of-fold predictions for stacking.
- Features are built on train + test together and only from columns, never from labels, so they are the same
  in validation and test.
- Validation labels are noisy and test labels are clean, so the out-of-fold score will read lower than the board.
  Keep two scores: all rows, and a cleaned view (rows where ring structure and label agree). Use the first one or
  two uploads to learn how the two map to the public board.

## Steps
1. **Baseline (sub1):** single-row LightGBM + phone-ownership features, month folds, class-balanced. Confirms the
   upload format and the validation-to-board gap.
2. **Connection features** (the main lever), all over train + test sorted by time:
   - receiver: distinct senders and money in over the last/next 10 min, 1 h, 6 h, 24 h; money sent on soon after;
   - phone: accounts per phone, payments per phone per minute, phone is not the sender's own, extra phone;
   - chains: receiver pays out 85–100% of the amount within minutes; position and length of the chain;
   - pairs: payments in the opposite direction; ₹1–10 received, then a large payment back (fake_cashback);
   - sender: REFUND claims in 7 / 30 days, repeat payments to lending merchants late in the evening,
     COLLECT just under a round limit, new phone + away from home city + drain of the balance;
   - merchant: round-number sales from the same few accounts; QR payee mismatch;
   - amounts: round, just under round, share of balance.
3. **Ring-level view:** group rows into events (same receiver burst, same phone burst, same chain, same
   sender-merchant run), add event-level statistics to every row, and smooth predictions inside an event.
4. **Label cleaning:** drop fraud-labelled rows that the out-of-fold models and the ring structure both call
   ordinary (`find_noisy.py`), and relabel legit rows inside confident rings, then retrain. Validation keeps all rows.
5. **Drift check on every feature:** adversarial train-vs-test AUC per feature; drop or re-scale what moves.
6. **Models:** LightGBM first, then XGBoost and CatBoost and seeds on Colab for variety; stack with the kit.
7. **Decision rule:** per-class biases tuned to macro-F1 (`postprocess.py`), cross-fitted, kept only if
   `compare.py` says the gain is real.
8. **Cross-fitted pseudo-labels on October** (test labels are clean and the rings adapted): `--pseudo`, teacher
   refreshed each round, echo check each time.
9. **Final:** last upload is the best-validated file; `reproduce.sh` complete; repo pushed only on "push";
   code link submitted well before 10:00 AM on 15 Nov.

## Timeline
- 13 Nov: steps 1–2, first uploads. Night: steps 3–4.
- 14 Nov: steps 5–8, one idea per upload.
- 15 Nov: nothing new after about 7:00 AM; final file and code link by 9:00 AM.

## Compute
Tabular data, so the laptop CPU carries most of it and the 4 GB GPU stays idle. Colab runs the heavier
XGBoost / CatBoost seeds from `upi_fraud_rings_colab.ipynb`.

## Findings after the first models (13 Nov, 3:30 AM)
- **Label noise is mostly random fraud labels on legit rows.** 36,619 fraud-labelled rows (2.3% of train, about 63%
  of all fraud labels) cannot be told from legit rows at all (AUC 0.50 with every feature, `ip_risk_score` included).
  The clean test will not have them, so true fraud is about 1.4% of rows, which is also what the models flag in test.
- The other direction is smaller: of rows the model is sure are fraud, about 13% are labelled legit (unreported) and
  about 14% carry another fraud type (mostly look-alike pairs: account_takeover / sim_swap, fake_cashback /
  phishing_collect, layering_chain / mule_account).
- So scores on the labels as given are capped near 0.47; what matters is the score on cleaned labels and the board.
  `clean_labels.py` writes cleaned labels; `LABELS=preds/labels_c1.npy` makes training, bias tuning and scoring use them.
- **October is half as dense:** 8,400 rows a day against 17,000, with the same accounts (rows per sender match a 50%
  thinning of September exactly). Rings are intact (chains of 4+ hops are as common per row as in September).
- velocity_bot: every real burst in train comes from an extra phone, whole-rupee amounts, e-commerce or gaming,
  about 4 s from login to payment. October has 7 such phones against 25 in September; no new bot pattern found yet.

## Findings, 13 Nov 6:45-7:20 AM
- **Density does not matter.** `sim_density.py`: train on July + August, score the September rows that survive a 50%
  thinning of ordinary rows. Full density throughout 0.8634; September thinned 0.8664; every month thinned 0.8645.
  All within noise, so training stays at full density.
- **The QR mismatch flag is a generator artefact, and October changes it.** In train an ordinary row has a flag value
  exactly when it is a QR payment, and never pays a person by QR. Rows that break either rule ("marked" rows) are fraud:
  87.8% carry a fraud label, which is what 13% unreported fraud leaves. They are a third of all real fraud rows.
  October keeps the artefact (613 marked rows per 100k against 650-680) but blanks the flag on 44% of its QR-to-person
  rows, a combination train never shows. `sub2.csv` calls 129 of October's 1,597 marked rows legit and `sub3.csv` 163
  (September: 0.4-3%). `features.py flag` gives the states explicitly, plus counts of marked rows in the same event;
  models then drop the raw column (`--drop qr_vpa_mismatch`).
- **Other October changes are small.** `ip_risk_score` no longer separates fraud (flagged rows 0.41 in September, 0.235
  in October, ordinary rows 0.20 / 0.22). Bots run more at night. Fewer merchant-side events (bots, collusion, refunds).
  No new category values, no hidden bursts: every multi-account phone burst in October is already flagged.
- **Raw ids must stay out of the features.** Fraud phones in train have ids 420033-428418; October's takeover and
  SIM-swap phones come from a new range above 473631. `phone_extra` (id >= 420000) still holds.
- **Wrong fraud types are whole events swapped between look-alike pairs** (about 28% between account_takeover and
  sim_swap), plus about 3% of rows at random. The model learns the right boundary anyway.
- Accounts and merchants all predate July, so account-age features only drift upward; left in for now.

## Cloud lane
`kaggle_run.py` pushes kit commands to private Kaggle CPU kernels (4 cores, 30 GB, no GPU quota) and fetches the
predictions. Data: private dataset `upi-fraud-rings-data` (the competition zip, `labels_c1.npy`, `thin_rows.npy`).
The code travels inside each kernel, so a push always runs the code as it is.

## Board lesson, 13 Nov 8:45 PM
sub5 (0.5038) and sub12 (0.5004) both scored below sub4 (0.5060). Both had biases tuned on the test-like labels t1 and
both flag more rows than sub4 (+39 and +83), mostly legit -> qr_swap and legit -> phishing_collect. The borderline
rows the t1 rule adds are wrong more often than validation says. sub13 = sub12's model tuned on cleaned labels like
sub4 (flags 12 fewer); sub14 = sub4 with the legit bias +2 (flags 95 fewer). sub14 recipe: add 2 to column legit of
log(flag_c1_thr probabilities), take the argmax.

## 14 Nov, 2:00 AM: sub15 = 0.4947
Type-corrected labels (c3) looked better on every validation view (+0.004 on t1) and lost 0.011 on the board. Every
change since sub4 that validation preferred has scored lower publicly (sub5 0.5038, sub12 0.5004, sub14 0.5012,
sub15 0.4947). Our validation cannot rank files that differ by ~100 rows; sub4 remains the best and the safe final.
The type swaps in train are probably how the generator labels these events, not noise to correct.
