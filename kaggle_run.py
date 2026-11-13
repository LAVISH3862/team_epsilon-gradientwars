"""Run kit commands on Kaggle (4 CPU cores, 30 GB RAM, no GPU quota used) and fetch the predictions back.
The code travels inside the kernel itself (a zip of *.py and kit/, base64), so a push always runs the code as it is
now; the data comes from the private dataset DATA (the competition zip plus preds/*.npy the jobs need).
  python kaggle_run.py push seeds "python features.py row conn ring" "python train_gbm.py --tag ring_s3 ..."
  python kaggle_run.py status seeds
  python kaggle_run.py fetch seeds          # copies the new preds/<tag>_{oof,test}.npy and the score lines here
Environment variables go in front of a command as usual: "LABELS=preds/labels_c1.npy python train_gbm.py ..."."""
import argparse, base64, glob, io, json, os, shutil, subprocess, sys, zipfile

USER, DATA = 'lavishxp', 'lavishxp/upi-fraud-rings-data'

TEMPLATE = r'''
import base64, glob, io, os, shutil, subprocess, time, zipfile
CODE = "__CODE__"
JOBS = __JOBS__
W, OUT = '/kaggle/working/upi', '/kaggle/working/out'
os.makedirs(W + '/data', exist_ok=True)
os.makedirs(W + '/preds', exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.chdir(W)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall(W)
names = ('train.csv', 'test.csv', 'accounts.csv', 'merchants.csv', 'sample_submission.csv')
for n in names:
    hits = glob.glob('/kaggle/input/**/' + n, recursive=True)
    if hits:
        os.symlink(hits[0], 'data/' + n)
if not os.path.exists('data/train.csv'):   # the dataset still holds the competition zip
    for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():
                if os.path.basename(m) in names:
                    open('data/' + os.path.basename(m), 'wb').write(f.read(m))
given = set()
for f in glob.glob('/kaggle/input/**/*.npy', recursive=True):
    shutil.copy(f, 'preds/')
    given.add(os.path.basename(f))
print('data:', sorted(os.listdir('data')), '| preds given:', sorted(given), flush=True)
ok = True
for j in JOBS:
    t0 = time.time()
    print('>>>', j, flush=True)
    r = subprocess.run(j, shell=True, env=dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE'))
    print('<<< exit %d after %.1f min' % (r.returncode, (time.time() - t0) / 60), flush=True)
    if r.returncode:
        ok = False
        break
for f in glob.glob('preds/*.npy'):
    if os.path.basename(f) not in given:
        shutil.copy(f, OUT)
for f in ('logs/scores.log', 'folds.csv'):
    if os.path.exists(f):
        shutil.copy(f, OUT)
os.chdir('/kaggle/working')
shutil.rmtree(W)                           # keep only out/ as the kernel's output
print('done' if ok else 'FAILED', sorted(os.listdir(OUT)), flush=True)
'''


def kaggle(*args, check=True):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    if check and r.returncode:
        raise SystemExit(r.returncode)
    return r.stdout + r.stderr


ap = argparse.ArgumentParser()
ap.add_argument('action', choices=['push', 'status', 'fetch', 'log'])
ap.add_argument('name')
ap.add_argument('jobs', nargs='*')
ap.add_argument('--gpu', action='store_true')
a = ap.parse_args()
slug, folder = f'upi-{a.name}', f'cloud/{a.name}'
if a.action == 'push':
    if not a.jobs:
        raise SystemExit('give at least one command to run')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob('*.py') + glob.glob('kit/*.py')):
            z.write(f)
    os.makedirs(folder, exist_ok=True)
    code = TEMPLATE.replace('__CODE__', base64.b64encode(buf.getvalue()).decode()).replace('__JOBS__', json.dumps(a.jobs, indent=1))
    open(f'{folder}/script.py', 'w', encoding='utf-8', newline='\n').write(code)
    json.dump({'id': f'{USER}/{slug}', 'title': slug, 'code_file': 'script.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': 'true', 'enable_gpu': 'true' if a.gpu else 'false',
               'enable_internet': 'false', 'dataset_sources': [DATA], 'competition_sources': [], 'kernel_sources': []},
              open(f'{folder}/kernel-metadata.json', 'w'), indent=1)
    open(f'{folder}/jobs.txt', 'w', encoding='utf-8', newline='\n').write('\n'.join(a.jobs) + '\n')
    kaggle('kernels', 'push', '-p', folder)
elif a.action == 'status':
    kaggle('kernels', 'status', f'{USER}/{slug}')
else:
    out = f'{folder}/out'
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    kaggle('kernels', 'output', f'{USER}/{slug}', '-p', out, check=False)
    logs = glob.glob(f'{out}/*.log')
    if a.action == 'log':
        for f in logs:
            print(open(f, encoding='utf-8', errors='replace').read()[-6000:])
        sys.exit()
    got = [f for f in glob.glob(f'{out}/**/*.npy', recursive=True)]
    for f in got:
        shutil.copy(f, 'preds/')
    for f in glob.glob(f'{out}/**/scores.log', recursive=True):
        lines = open(f, encoding='utf-8').read()
        open('logs/scores.log', 'a', encoding='utf-8').write(''.join(f'[kaggle {a.name}] {l}\n' for l in lines.splitlines()))
        print(lines)
    print(f'copied {len(got)} prediction files to preds/:', sorted(os.path.basename(f) for f in got))
