"""Compare two tags before an upload: out-of-fold score and held-out score (make_folds.py --holdout), each with a paired
bootstrap (is the gain real or noise?), per-class change, and which test labels move. The verdict comes from the
held-out rows when there are any: out-of-fold scores flatter pseudo-labels and noise cleaning, the held-out rows don't.
  python compare.py old_tag new_tag
  python compare.py old_tag new_tag --boot 0     # skip the bootstrap
  python compare.py old_tag new_tag --gate       # exit code 1 unless the gain is real (for scripts)
  python compare.py --echo pl_tag teacher_tag    # does a model trained on pseudo-labels just echo its teacher?"""
import argparse
from kit.common import y, classes, load_preds, cv_rows, holdout_rows, bootstrap, score
from kit import metrics
from kit.pseudo import echo
import numpy as np, pandas as pd
import comp

ap = argparse.ArgumentParser()
ap.add_argument('a')
ap.add_argument('b')
ap.add_argument('--echo', action='store_true', help='a = model trained on pseudo-labels, b = its teacher')
ap.add_argument('--thr', type=float, default=0.9)
ap.add_argument('--boot', type=int, default=1000, help='paired bootstrap draws (0 to skip)')
ap.add_argument('--min-gain', type=float, default=0.0, help='a real gain must also be at least this big')
ap.add_argument('--gate', action='store_true', help='exit code 1 unless the verdict is a real gain')
a = ap.parse_args()
yy = y()
(Oa, Ta), (Ob, Tb) = load_preds(a.a), load_preds(a.b)
if a.echo:
    if comp.TASK == 'regression':
        raise SystemExit('the echo check needs class probabilities')
    agree, honest = echo(a.a, a.b, a.thr)
    print(f'{a.a} agrees with {a.b} on {agree:.2%} of the teacher-confident test rows; its own confident validation rows '
          f'are {honest:.2%} right. Echo {agree - honest:+.2%}: '
          + ('it copies its teacher. Retrain with cross-fitted pseudo-labels or keep it out of stacks.'
             if agree - honest > 0.01 else 'no meaningful echo.'))
    raise SystemExit
verdict = None
for name, rows in (('out-of-fold', cv_rows()), ('held-out', holdout_rows())):
    if not len(rows):
        continue
    sa, sb = score(Oa[rows], rows), score(Ob[rows], rows)
    line = f'{name} {comp.METRIC} ({len(rows)} rows): {a.a} {sa:.4f} -> {a.b} {sb:.4f} ({sb - sa:+.4f})'
    if a.boot:
        change, (lo, hi), p = bootstrap(Oa, Ob, rows, a.boot)
        big = abs(change) >= a.min_gain
        verdict = ('a real gain.' if p >= 0.95 and big else 'a real loss.' if p <= 0.05 else
                   'within noise, not proof either way.')
        line += (f'; paired bootstrap{" by group" if comp.GROUP else ""}: 95% CI [{lo:+.4f}, {hi:+.4f}], {a.b} better in '
                 f'{p:.0%} of draws: {verdict}')
    print(line)
if verdict:
    print(f'verdict ({"held-out rows" if len(holdout_rows()) else "out-of-fold"}): {verdict}')
if comp.TASK == 'regression':
    print(f'test predictions: correlation {np.corrcoef(Ta, Tb)[0, 1]:.4f}, mean absolute change {np.abs(Ta - Tb).mean():.4f}')
else:
    cv = cv_rows()
    fa, fb = metrics.per_class_f1(yy[cv], Oa[cv]), metrics.per_class_f1(yy[cv], Ob[cv])
    print(pd.DataFrame({a.a: fa, a.b: fb, 'change': fb - fa}, index=classes()).round(3).to_string())
    la, lb, C = Ta.argmax(1), Tb.argmax(1), classes()
    ch = la != lb
    print(f'test labels changed: {ch.sum()} of {len(la)}')
    if ch.any():
        print(pd.Series([f'{C[i]} -> {C[j]}' for i, j in zip(la[ch], lb[ch])]).value_counts().head(10).to_string())
if a.gate and verdict != 'a real gain.':
    raise SystemExit(1)
