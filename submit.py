"""Turn a tag into the next upload file, submissions/subN.csv (plain names: the file never says what's inside), after
checking it against sample_submission. submissions/log.csv records which tag, validation and held-out score each subN is.
  python submit.py stack_pl --note "cross-fitted stack, teacher round 2"
  python submit.py --lb sub3 0.9133          # record the public score once it shows
  python submit.py stack_pl --name sub7      # in reproduce.sh: rebuild an uploaded file under its own name"""
import argparse, glob, os, re
from kit.common import test, classes, load_preds, scores
import numpy as np, pandas as pd
import comp

ap = argparse.ArgumentParser()
ap.add_argument('tag', nargs='?')
ap.add_argument('--note', default='')
ap.add_argument('--name', default='')
ap.add_argument('--lb', nargs=2, metavar=('SUB', 'SCORE'))
a = ap.parse_args()
LOG = 'submissions/log.csv'
log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=['file', 'time', 'tag', 'val', 'hold', 'changed', 'public', 'note'])
if a.lb:
    log.loc[log.file == a.lb[0], 'public'] = float(a.lb[1])
    log.to_csv(LOG, index=False)
    raise SystemExit(log.tail(10).to_string(index=False))
if not a.tag:
    raise SystemExit('usage: python submit.py TAG [--note ...]')

oof, P = load_preds(a.tag)
ss, te = comp.load_sample(), test()
cols = [c for c in ss.columns if c != comp.ID]
if comp.ID in te:
    order = pd.Series(np.arange(len(te)), index=te[comp.ID].values).reindex(ss[comp.ID].values).values
    if np.isnan(order).any():
        raise SystemExit('some sample_submission ids are not in the test set')
    order = order.astype(int)
elif len(te) == len(ss):
    order = np.arange(len(ss))  # no id column in test: assume the sample's row order
else:
    raise SystemExit('test and sample_submission differ in length')
sub = ss[[comp.ID]].copy()
if comp.TASK == 'regression':
    sub[cols[0]] = P[order]
elif comp.SUBMIT == 'label':
    sub[cols[0]] = classes()[P[order].argmax(1)]
elif len(cols) == 1:
    sub[cols[0]] = P[order, -1]  # binary: probability of the second (positive) class
else:
    for k, c in enumerate(classes()):
        sub[str(c)] = P[order, k]
    missing = set(map(str, cols)) - set(sub.columns)
    if missing:
        raise SystemExit(f'sample_submission columns {missing} are not class names')
    sub = sub[[comp.ID] + cols]
if len(sub) != len(ss) or sub.isna().any().any():
    raise SystemExit('submission has the wrong length or empty values')

nums = [int(m.group(1)) for f in glob.glob('submissions/sub*.csv') if (m := re.fullmatch(r'sub(\d+)\.csv', os.path.basename(f)))]
name = a.name or f'sub{max(nums, default=0) + 1}'
prev = log.file.iloc[-1] if len(log) else None
changed = ''
if prev and prev != name and os.path.exists(f'submissions/{prev}.csv') and comp.TASK != 'regression' and comp.SUBMIT == 'label':
    old = pd.read_csv(f'submissions/{prev}.csv')
    changed = int((old[cols[0]].astype(str).values != sub[cols[0]].astype(str).values).sum())
sub.to_csv(f'submissions/{name}.csv', index=False)
val, hold = scores(oof)
row = dict(file=name, time=pd.Timestamp.now().strftime('%m-%d %H:%M'), tag=a.tag, val=round(val, 4),
           hold='' if hold is None else round(hold, 4), changed=changed, public='', note=a.note)
new = pd.DataFrame([row])
(pd.concat([log, new], ignore_index=True) if len(log) else new).to_csv(LOG, index=False)
print(f'wrote submissions/{name}.csv: {len(sub)} rows, validation {comp.METRIC} {val:.4f}'
      + ('' if hold is None else f', held-out {hold:.4f}')
      + (f', {changed} labels changed vs {prev}' if changed != '' else ''))
if comp.TASK != 'regression' and comp.SUBMIT == 'label':
    print(sub[cols[0]].value_counts().to_string())
