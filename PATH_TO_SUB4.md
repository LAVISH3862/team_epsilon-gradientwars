# How we got to sub4 (public 0.5060)

Everything from opening the data to the file that is still our best. Times are 13 Nov 2026, IST.
sub4 was written at 7:29 AM and uploaded at 10:01 AM, the moment the board opened.

---

## Step 1. Setup and first look (1:55–2:10 AM)

- Copied the competition kit into `upi-fraud-rings`, unzipped the data, read `PROBLEM_STATEMENT.md` and the starter
  notebook, and wrote the rules (macro-F1, 10 uploads a day, 30/70 public/private split, code link by 15 Nov
  10:00 AM) into `comp.py`.
- **Folds by month** (July, August, September): test is the next month and a ring's rows sit close in time, so
  random folds would leak rings. September is the honest check: models trained on July and August predict it.
- **What the data showed:**
  - 96.3% legit; 11 fraud types at 0.13–0.65% each.
  - `app_version` is 3.0–4.8 in train and only 5.0–5.2 in test, so it was left out from the start.
  - `ip_risk_score` is distributed differently in test, so it was left out too.
  - An account's own phone carries its number (`AC377784` ↔ `DV0377784`); phones numbered 420000+ are extra
    phones, used by 37% of velocity_bot rows and 25% of account_takeover rows against 3% of legit.
  - Train runs into early October (147 rows) and test into early November (33 rows).

## Step 2. Baseline: single-row features (2:10–3:00 AM)

- `features.py row`: 43 features from each row on its own: amount patterns (whole rupees, just under a round
  limit, share of balance), time, payment type, merchant category, SIM / phone / account ages, PIN retries,
  login time, and whose phone it is.
- LightGBM, class-balanced, legit capped at 150,000 rows per fold for speed.
- `postprocess.py`: per-class biases tuned for macro-F1 (cross-fitted). This alone lifted the score from 0.16
  to 0.39, because macro-F1 rewards calling rare types more readily.
- **Result:** out-of-fold macro-F1 **0.3933** (written as sub1, never uploaded).

## Step 3. Connection features (2:15–3:05 AM)

- `features.py conn`: 102 features linking each row to other rows, built over train and test together
  (allowed by the rules), looking both back and ahead in time:
  - **Receiver:** how many payments and different senders in 10 min / 1 h / 6 h / 24 h; new-payee share;
    round-amount share; QR mismatches seen.
  - **Money moving on:** does the receiver pay out soon after, how much, how fast.
  - **Sender:** bursts, refunds in 7 / 30 days, collect requests, lending payments, money received just before.
  - **Chains:** hops where most of the money moves on within the hour, and how long the chain is.
  - **Phone:** payments per minute / hour / day, how many accounts use it.
  - **Pair:** repeats between the same two accounts, and payments the other way round (the ₹1–10 bait of
    fake cashback).
- **Result:** **0.4685** (written as sub2, never uploaded). Biggest gains: refund_abuse, mule_account,
  fake_cashback, merchant_collusion and layering_chain each about +0.14 to +0.16.

## Step 4. The label-noise discovery and cleaning (3:10–3:45 AM)

- 79% of fraud-labelled rows had no other fraud row on the same sender, phone or receiver that day.
- A model could not tell 36,227 fraud-labelled rows (2.3% of train) from legit rows at all (AUC 0.50 with every
  feature), while ring fraud is 1.5% of rows. The problem statement explains both: "some fraud looks completely
  ordinary", and training labels are imperfect.
- Inside real rings, about 13% of rows were labelled legit (unreported fraud), and whole events were sometimes
  filed under a look-alike type.
- `clean_labels.py` (labels `c1`): fraud labels on rows the model sees as ordinary were treated as legit for
  training, and legit-labelled rows the model was sure were fraud took the model's type. Always from
  out-of-fold predictions, so no row cleaned its own label.
- **Result:** the cleaned-label view rose to 0.86, and the labels-as-given score held at 0.47: the model now
  learned the real rings instead of noise.

## Step 5. Ring-level features (3:45–6:35 AM)

- `features.py ring`: 89 features describing the event each row belongs to: the sender's session (2-hour
  gaps), the receiver's session (6-hour gaps), the phone's burst (10-minute gaps), this account on this phone,
  and this sender–receiver pair. Means, minimums, maximums and spreads of amount, SIM age, phone age, login
  time, new payees, night payments, whole amounts and refunds over each event.
- **Result:** 0.8615 on cleaned labels (written as sub3, never uploaded): a small gain, mainly for
  account_takeover and merchant_collusion.

## Step 6. Checking October for traps (6:40–7:10 AM)

- **Density:** October has half the payments per day of the training months. A test (`sim_density.py`)
  training and scoring at full and half density showed no effect (0.863 / 0.866 / 0.865), so training stayed
  at full density.
- **The QR mismatch flag (the find that made sub4):**
  - In train, an ordinary row has a QR flag value exactly when it is a QR payment, and never pays a person by
    QR. Rows breaking either rule are fraud 99.4% of the time (a third of all real fraud).
  - October blanks the flag on 44% of its QR-to-person payments, a combination train never shows.
  - Because the model read the raw flag, sub3 called 163 of October's 1,597 such near-certain fraud rows legit.

## Step 7. sub4: the flag fix (7:10–7:29 AM)

- `features.py flag`: 23 features turning the QR flag into explicit states (value without a QR payment, QR
  payment without a value, QR to a person) and counting such "marked" rows around each row (same receiver,
  sender, phone and pair). The raw `qr_vpa_mismatch` column was dropped (`--drop qr_vpa_mismatch`).
- Model: LightGBM on row + conn + ring + flag (257 features), trained on cleaned labels `c1`, class-balanced,
  legit capped at 150,000, biases tuned on cleaned labels.
- **Result:**
  - Validation unchanged (0.8612 cleaned, 0.4764 as given), as expected: the fix matters only in October.
  - In October, near-certain fraud rows called legit fell from **163 to 4**, and the qr_swap share matched
    September (0.201% vs 0.202%).
  - Flags 3,787 test rows (1.45%) as fraud.
- **Uploaded at 10:01 AM: public 0.5060.**

---

## The recipe in one place (from `reproduce.sh`)

```
python make_folds.py
python features.py row
python features.py conn
python train_gbm.py --tag conn --feats row+conn --balanced --cap 150000 --seeds 1
python clean_labels.py conn --name c1
python features.py ring
python features.py flag
LABELS=preds/labels_c1.npy python train_gbm.py --tag flag_c1 --feats row+conn+ring+flag --drop qr_vpa_mismatch --balanced --cap 150000 --seeds 1
LABELS=preds/labels_c1.npy python postprocess.py flag_c1 --tag flag_c1_thr --grid 8 --step 0.25
LABELS=preds/labels_c1.npy python submit.py flag_c1_thr --name sub4
```
