"""Cross-fitted pseudo-labels: the biggest lesson of Hack4Hack 2026.

Training on confident test predictions teaches a model what the test data looks like. But a model must never
predict a test row whose own pseudo-label it trained on: it just hands the teacher's answer back. Our final stacker
agreed with its old teacher on 100.00% of those rows, so a night of better models barely moved the score; cross-
fitting gave +0.013 in an hour. Fix: split the test rows into N_FOLDS parts and pair part k with training fold k.
The fold-k model trains on the pseudo-labels of the other parts and is the only model that predicts part k.
Refresh the teacher (the latest best tag) every round; with cross-fitting each round helped (+0.004 per round).
teacher_loop.py chains the rounds. Soft labels (--soft) train on the teacher's probabilities instead of its argmax, so
a row the teacher is unsure about teaches less. A regression teacher's predictions label every test row.
Out-of-fold scores of pseudo-label models are optimistic (the teacher saw the validation folds): judge them on the
held-out rows (make_folds.py --holdout), which no teacher ever trained on."""
import numpy as np
import comp


class PseudoLabels:
    def __init__(self, teacher_test_preds, thr=0.9, seed=0, soft=False):
        P = np.asarray(teacher_test_preds)
        self.part = np.random.default_rng(seed).integers(0, comp.N_FOLDS, len(P))
        if P.ndim == 1:  # regression: every test row, labelled with the teacher's prediction
            self.rows, self.labels = np.arange(len(P)), P
        else:
            self.rows = np.where(P.max(1) >= thr)[0]
            self.labels = P[self.rows] if soft else P.argmax(1)[self.rows]
        kind = 'targets' if P.ndim == 1 else 'soft labels' if soft else 'labels'
        print(f'pseudo-{kind}: {len(self.rows)} of {len(P)} test rows' + ('' if P.ndim == 1 else f' at thr {thr}')
              + f', cross-fitted (parts seed {seed})', flush=True)

    def for_fold(self, k):
        """(test rows to train on, their pseudo-labels, test rows that this fold's model predicts)."""
        m = self.part[self.rows] != k
        return self.rows[m], self.labels[m], np.where(self.part == k)[0]


def echo(student, teacher, thr=0.9):
    """Does a model trained on pseudo-labels copy its teacher? Returns (its agreement with the teacher on the teacher-
    confident test rows, its accuracy on its own confident validation rows); a gap above 1% means it echoes."""
    from kit.common import load_preds, cv_rows, y
    (Os, Ts), (_, Tt) = load_preds(student), load_preds(teacher)
    cv = cv_rows()
    tc, oc = Tt.max(1) >= thr, cv[Os[cv].max(1) >= thr]
    return float(np.mean(Ts.argmax(1)[tc] == Tt.argmax(1)[tc])), float(np.mean(Os.argmax(1)[oc] == y()[oc]))


def expand(rows, labels, min_p=0.01):
    """For models that only take hard labels (GBMs): soft labels become one copy of the row per likely class, weighted
    by its probability. Hard labels and regression targets pass through with weight 1."""
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return rows, labels, np.ones(len(rows))
    r, c = np.nonzero(labels >= min_p)
    w = labels[r, c]
    return rows[r], c, w / np.bincount(r, weights=w, minlength=len(labels))[r]
