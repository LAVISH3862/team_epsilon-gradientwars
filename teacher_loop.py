"""Chained, cross-fitted pseudo-label rounds in one command. Each round, every student trains on the current teacher's
pseudo-labels (cross-fitted: no model predicts a test row whose pseudo-label it saw), the combiner builds the next teacher
from the students, and the loop prints the round's scores, each student's echo check and whether the new teacher beat
the old one. The 4th-place team in Hack4Hack 2026 ran ~10 rounds; each of our 3 rounds was worth about +0.004.
In a command, {r} is the round number and {prev} the current teacher; students get --pseudo <teacher> appended.
  python teacher_loop.py --teacher stack --rounds 4 \\
      --student "train_gbm.py --tag gbm_r{r} --balanced --soft --thr 0.5" \\
      --student "train_gbm.py --tag xgb_r{r} --model xgb --balanced --soft --thr 0.5" \\
      --combine "blend.py --tag teacher_r{r} gbm_r{r} xgb_r{r}"
Judge the rounds on the held-out rows (make_folds.py --holdout): out-of-fold scores of pseudo-label models flatter,
because the teacher saw the validation folds. A command whose tag already has predictions is skipped, so a rerun resumes
where a crash or a lost Colab session stopped. Students run one at a time (one GPU job at a time)."""
import argparse, os, re, shlex, subprocess, sys, time
from kit.common import load_preds, scores, cv_rows, holdout_rows, bootstrap
from kit.pseudo import echo
from kit import metrics
import pandas as pd
import comp

ap = argparse.ArgumentParser()
ap.add_argument('--teacher', required=True, help='the round-0 teacher tag')
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--start', type=int, default=1, help='number of the first round (to continue an earlier loop)')
ap.add_argument('--student', action='append', required=True, help='a training command; repeat for every student')
ap.add_argument('--combine', required=True, help='the command that builds the next teacher from the students')
ap.add_argument('--patience', type=int, default=2, help='stop after this many rounds without a better teacher')
a = ap.parse_args()
judge = 'held-out' if len(holdout_rows()) else 'out-of-fold'
rows = holdout_rows() if len(holdout_rows()) else cv_rows()


def run(cmd):
    """Run `python cmd` unless its tag already has predictions; its output goes to logs/<tag>.log."""
    tag = re.search(r'--tag\s+(\S+)', cmd).group(1)
    if os.path.exists(f'preds/{tag}_oof.npy') and os.path.exists(f'preds/{tag}_test.npy'):
        print(f'  {tag}: already done, skipped', flush=True)
        return tag
    t0 = time.time()
    with open(f'logs/{tag}.log', 'w') as f:
        code = subprocess.call([sys.executable] + shlex.split(cmd), stdout=f, stderr=subprocess.STDOUT)
    if code:
        raise SystemExit(f'{tag} failed (exit {code}): see logs/{tag}.log')
    print(f'  {open(f"logs/{tag}.log").read().strip().splitlines()[-1]}  [{time.time() - t0:.0f}s]', flush=True)
    return tag


def judged(tag):
    cv, ho = scores(load_preds(tag)[0])
    return cv, ho, (cv if ho is None else ho)


teacher = a.teacher
cv, ho, best = judged(teacher)
best_tag, stale, table = teacher, 0, [dict(round=0, teacher=teacher, cv=cv, held_out=ho)]
print(f'round 0: teacher {teacher}, {judge} {comp.METRIC} {best:.4f}', flush=True)
for r in range(a.start, a.start + a.rounds):
    t0 = time.time()
    fill = lambda c: c.replace('{r}', str(r)).replace('{prev}', teacher)
    print(f'round {r}: teacher {teacher}', flush=True)
    students = [run(fill(c) + ('' if '--pseudo' in c else f' --pseudo {teacher}')) for c in a.student]
    new = run(fill(a.combine))
    echoes = {}
    if comp.TASK != 'regression':
        for s in students:
            agree, honest = echo(s, teacher)
            echoes[s] = agree - honest
            print(f'  echo {s}: {agree - honest:+.2%}' + ('  <- copies its teacher: keep it out of the combiner'
                                                          if agree - honest > 0.01 else ''), flush=True)
    cv, ho, js = judged(new)
    change, (lo, hi), p = bootstrap(load_preds(teacher)[0], load_preds(new)[0], rows, 500)
    verdict = 'a real gain' if p >= 0.95 else 'a real loss' if p <= 0.05 else 'within noise'
    print(f'  {new}: {judge} {comp.METRIC} {js:.4f} ({change:+.4f} vs {teacher}, 95% CI [{lo:+.4f}, {hi:+.4f}]: '
          f'{verdict}); round took {time.time() - t0:.0f}s', flush=True)
    table.append(dict(round=r, teacher=new, cv=cv, held_out=ho, change=change, verdict=verdict,
                      max_echo=max(echoes.values()) if echoes else None, seconds=round(time.time() - t0)))
    if metrics.better(comp.METRIC, js, best):
        best, best_tag, stale = js, new, 0
    else:
        stale += 1
    teacher = new
    if stale >= a.patience:
        print(f'stopping: no better teacher in {stale} rounds', flush=True)
        break
summary = pd.DataFrame(table).round(4).to_string(index=False)
print(summary + f'\nbest teacher by {judge} {comp.METRIC}: {best_tag} ({best:.4f})')
with open('logs/teacher_loop.log', 'a') as f:
    f.write(f'{pd.Timestamp.now():%m-%d %H:%M}\n{summary}\nbest: {best_tag}\n\n')
