"""Pretrained CNN on comp.load_images: a torchvision model (--arch efficientnet_b0) or any timm model
(--arch timm:<name>, normalised with that model's own mean/std), trained on the fixed folds, optionally with cross-fitted
pseudo-labels. Keeps each fold's best epoch (chosen on that fold, so out-of-fold scores are slightly optimistic) and
predicts with flip TTA when comp.HFLIP_OK. Writes models/<tag>_f<k>.pt, preds/<tag>_f<k>_*.npy, and
preds/<tag>_{oof,test}.npy once every fold exists. Use a new tag for every new setting, or old fold files will mix in.
Held-out rows (make_folds.py --holdout) are never trained on; every fold model predicts them through comp.simulate_images.
  python train_cnn.py --tag effb0 --arch efficientnet_b0 --size 224 --epochs 8
  python train_cnn.py --tag cnxt --arch timm:convnext_tiny.fb_in22k_ft_in1k --ema 0.999 --head-lr-mult 10
  python train_cnn.py --tag effb0_pl --arch efficientnet_b0 --pseudo stack --thr 0.9
  python train_cnn.py --tag effb0_spl --arch efficientnet_b0 --pseudo stack --soft --thr 0.5   # teacher's probabilities
  python train_cnn.py --tag effb0_c1 --arch efficientnet_b0 --exclude preds/noisy_r1.npy   # suspected wrong labels out
  python train_cnn.py --tag effb0 --folds 0,1          # some folds now, the rest later; merges when all exist
timm names: python -c "import timm; print(timm.list_models('*convnext*', pretrained=True))"; e.g. efficientnet_b2.ra_in1k,
tf_efficientnetv2_s.in21k_ft_in1k, convnext_tiny.fb_in22k_ft_in1k, regnety_016.tv2_in1k, resnet50d.ra2_in1k."""
import argparse, os, time
from kit.common import y, folds, classes, test, save_preds, report, seed_all, SIM_SEED
from kit import metrics
from kit.pseudo import PseudoLabels
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from PIL import Image
import comp

MEAN = (0.485, 0.456, 0.406)   # ImageNet, what torchvision weights expect
STD = (0.229, 0.224, 0.225)


class Images(torch.utils.data.Dataset):
    """Rows idx of an image source (uint8 array or list of file paths) as uint8 (C, H, W) tensors, with labels
    (class numbers, or probability rows for soft labels)."""
    def __init__(self, src, idx, labels=None, size=224):
        self.src, self.idx, self.labels, self.size = src, np.asarray(idx), labels, size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        if isinstance(self.src, np.ndarray):
            x = np.asarray(self.src[j])
        else:
            x = np.asarray(Image.open(self.src[j]).convert('RGB').resize((self.size, self.size)))
        x = torch.from_numpy(np.ascontiguousarray(x if x.ndim == 3 else x[..., None])).permute(2, 0, 1)
        if self.labels is None:
            return x, -1
        return x, torch.as_tensor(self.labels[i], dtype=torch.float32) if self.labels.ndim == 2 else int(self.labels[i])


def as_array(src, idx, size):
    """Rows idx of an image source as one uint8 array (N, H, W[, 3]), e.g. to pass them through comp.simulate_images."""
    if isinstance(src, np.ndarray):
        return np.asarray(src[idx])
    return np.stack([np.asarray(Image.open(src[j]).convert('RGB').resize((size, size))) for j in idx])


def build(arch, n, pretrained):
    if arch.startswith('timm:'):                                                        # e.g. timm:legacy_xception.tf_in1k
        import timm
        return timm.create_model(arch[5:], pretrained=pretrained, num_classes=n)
    m = getattr(torchvision.models, arch)(weights='DEFAULT' if pretrained else None)
    if hasattr(m, 'fc'): m.fc = nn.Linear(m.fc.in_features, n)                        # resnet
    elif hasattr(m, 'heads'): m.heads.head = nn.Linear(m.heads.head.in_features, n)   # vit
    else: m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)               # efficientnet, convnext, mobilenet
    return m


def head_of(m):
    """The freshly initialised classification layer, which starts from scratch and can take a higher learning rate."""
    if hasattr(m, 'get_classifier'): return m.get_classifier()                         # timm
    if hasattr(m, 'fc'): return m.fc
    if hasattr(m, 'heads'): return m.heads.head
    return m.classifier[-1]


def norm_of(m):
    """The input mean and std the backbone was pretrained with: timm models carry their own (Xception and many ViTs use
    0.5, not ImageNet's), torchvision models use ImageNet's."""
    cfg = getattr(m, 'pretrained_cfg', None) or {}
    return [torch.tensor(cfg.get(k, d)).view(1, 3, 1, 1) for k, d in (('mean', MEAN), ('std', STD))]


def merge(tag, n_cls, n_test):
    """Combine the per-fold files into preds/<tag>_{oof,test}.npy once every fold has run."""
    missing = [k for k in range(comp.N_FOLDS) if not os.path.exists(f'preds/{tag}_f{k}_test.npy')]
    if missing:
        print(f'folds {missing} not run yet; preds/{tag}_oof.npy is written once they are'); return
    oof, pt, cnt = np.zeros((len(y()), n_cls), np.float32), np.zeros((n_test, n_cls), np.float32), np.zeros(n_test)
    hold = np.where(folds() < 0)[0]
    for k in range(comp.N_FOLDS):
        oof[np.load(f'preds/{tag}_f{k}_val_idx.npy')] = np.load(f'preds/{tag}_f{k}_val.npy')
        ti = np.load(f'preds/{tag}_f{k}_test_idx.npy')
        pt[ti] += np.load(f'preds/{tag}_f{k}_test.npy'); cnt[ti] += 1
        if len(hold):
            oof[hold] += np.load(f'preds/{tag}_f{k}_hold.npy') / comp.N_FOLDS
    save_preds(tag, oof, pt / cnt[:, None])
    report(tag, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--arch', default='efficientnet_b0', help='torchvision model (resnet34, efficientnet_b0-b3, convnext_tiny...) or timm:<name>')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--smooth', type=float, default=0.05, help='label smoothing')
    ap.add_argument('--folds', default='', help='comma-separated folds to run (default: all)')
    ap.add_argument('--pseudo', default='', help='teacher tag: cross-fitted pseudo-labels from preds/<teacher>_test.npy')
    ap.add_argument('--thr', type=float, default=0.9)
    ap.add_argument('--soft', action='store_true', help="train on the teacher's probabilities, not its argmax")
    ap.add_argument('--parts-seed', type=int, default=0, help='how test rows are split into parts; average a few runs')
    ap.add_argument('--workers', type=int, default=2, help='loader workers for image files (arrays load in-process)')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--exclude', default='', help='.npy of training rows to leave out of training (find_noisy.py); every row is still validated')
    ap.add_argument('--ema', type=float, default=0, help='weight averaging (timm ModelEmaV3 with warmup), e.g. 0.999; scores and saves the averaged model')
    ap.add_argument('--head-lr-mult', type=float, default=1, help='learning-rate multiplier for the new classification layer, e.g. 10')
    ap.add_argument('--seed', type=int, default=comp.SEED, help='a new seed (and tag) per run, then average the tags: blend.py --equal')
    ap.add_argument('--device', default='', help='cuda or cpu (default: cuda when available)')
    a = ap.parse_args()
    seed_all(a.seed)
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu'); cuda = dev == 'cuda'
    if not cuda:  # keep a CPU run off the GPU (another job may be training there): torch would open a CUDA context
        torch.cuda.is_available = lambda: False           # for the optimiser's graph-capture check
        torch.autograd.set_multithreading_enabled(False)  # and for autograd's per-device threads
    torch.backends.cudnn.benchmark = True
    Xtr, Xte = comp.load_images('train'), comp.load_images('test')
    yy, fo, n_cls, n_test = y(), folds(), len(classes()), len(test())
    excl = np.load(a.exclude) if a.exclude else np.array([], int)
    mean = std = None  # set from each fold's model: timm backbones have their own
    workers = 0 if isinstance(Xtr, np.ndarray) else a.workers
    pl = PseudoLabels(np.load(f'preds/{a.pseudo}_test.npy'), a.thr, a.parts_seed, a.soft) if a.pseudo else None
    hold = np.where(fo < 0)[0]  # held-out rows: never trained on, predicted like test rows, through the test simulator
    Xh = getattr(comp, 'simulate_images', lambda X, seed: X)(as_array(Xtr, hold, a.size), SIM_SEED) if len(hold) else None

    def loader(ds, shuffle):
        return torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=shuffle, num_workers=workers, pin_memory=cuda,
                                           drop_last=shuffle and len(ds) > a.bs, persistent_workers=workers > 0)

    def prep(x, augment):
        x = x.to(dev, non_blocking=True).float() / 255
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        if x.shape[-2:] != (a.size, a.size): x = F.interpolate(x, size=(a.size, a.size), mode='bilinear', align_corners=False)
        if augment:
            if comp.HFLIP_OK:
                f = torch.rand(len(x), device=dev) < 0.5
                x = torch.where(f[:, None, None, None], x.flip(-1), x)
            x = x * (1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=dev) - 0.5))  # brightness jitter
        return (x - mean) / std

    @torch.no_grad()
    def predict(model, src, idx):
        model.eval(); out = []
        for x, _ in loader(Images(src, idx, size=a.size), False):
            x = prep(x, False)
            with torch.autocast(device_type=dev, enabled=cuda):
                p = model(x).float().softmax(1)
                if comp.HFLIP_OK: p = (p + model(x.flip(-1)).float().softmax(1)) / 2  # flip TTA
            out.append(p.cpu().numpy())
        return np.concatenate(out)

    for k in [int(f) for f in a.folds.split(',')] if a.folds else range(comp.N_FOLDS):
        tr, va = np.setdiff1d(np.where((fo >= 0) & (fo != k))[0], excl), np.where(fo == k)[0]
        ds, rows = Images(Xtr, tr, np.eye(n_cls, dtype=np.float32)[yy[tr]] if a.soft else yy[tr], a.size), np.arange(n_test)
        if pl:
            r, lab, rows = pl.for_fold(k)
            ds = torch.utils.data.ConcatDataset([ds, Images(Xte, r, lab, a.size)])
        dl = loader(ds, True)
        model = build(a.arch, n_cls, not a.no_pretrained).to(dev)
        mean, std = (t.to(dev) for t in norm_of(model))
        head = {id(p) for p in head_of(model).parameters()}
        groups = [{'params': [p for p in model.parameters() if id(p) not in head], 'lr': a.lr},
                  {'params': [p for p in model.parameters() if id(p) in head], 'lr': a.lr * a.head_lr_mult}]
        opt = torch.optim.AdamW(groups, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=a.epochs * len(dl), pct_start=0.1)
        scaler = torch.amp.GradScaler('cuda', enabled=cuda) if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=cuda)
        ema, step, best = None, 0, None
        if a.ema:
            from timm.utils import ModelEmaV3
            ema = ModelEmaV3(model, decay=a.ema, use_warmup=True)
        for ep in range(a.epochs):
            model.train(); t0, tot = time.time(), 0.0
            for x, t in dl:
                x, t = prep(x, True), t.to(dev)
                with torch.autocast(device_type=dev, enabled=cuda):
                    loss = F.cross_entropy(model(x).float(), t, label_smoothing=a.smooth)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                step += 1
                if ema:
                    ema.update(model, step=step)
                tot += loss.item()
            scored = ema.module if ema else model
            pv = predict(scored, Xtr, va); s = metrics.score(comp.METRIC, yy[va], pv)
            print(f'fold {k} ep {ep} loss {tot / len(dl):.4f} val {comp.METRIC} {s:.4f}{" (EMA)" if ema else ""} {time.time() - t0:.0f}s', flush=True)
            if best is None or metrics.better(comp.METRIC, s, best):
                best = s
                torch.save(scored.state_dict(), f'models/{a.tag}_f{k}.pt')
                np.save(f'preds/{a.tag}_f{k}_val.npy', pv); np.save(f'preds/{a.tag}_f{k}_val_idx.npy', va)
        model.load_state_dict(torch.load(f'models/{a.tag}_f{k}.pt', map_location=dev))
        np.save(f'preds/{a.tag}_f{k}_test.npy', predict(model, Xte, rows)); np.save(f'preds/{a.tag}_f{k}_test_idx.npy', rows)
        if len(hold):
            np.save(f'preds/{a.tag}_f{k}_hold.npy', predict(model, Xh, np.arange(len(hold))))
        print(f'fold {k} best {comp.METRIC} {best:.4f}', flush=True)
        del model, opt, ema, scored
        if cuda: torch.cuda.empty_cache()
    merge(a.tag, n_cls, n_test)


if __name__ == '__main__':  # required on Windows: loader workers re-import this file
    main()
