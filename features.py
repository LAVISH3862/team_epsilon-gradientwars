"""Feature blocks, built over train + test together (connections between rows may use test rows; labels are never
used). Each block is cached as data/feat_<block>.parquet, indexed by transaction id, and picked by comp.BLOCKS
(env FEATS=row+recv or train_gbm.py --feats row+recv).
  python features.py row          # single-row features + phone ownership (the baseline)
Left out on purpose: app_version (3.0-4.8 in train, only 5.0-5.2 in test) and ip_risk_score (its distribution changes
in October)."""
import sys, time
import numpy as np, pandas as pd
import comp


def both():
    """Train and test stacked, with parsed time, sorted as loaded (train rows first)."""
    tr, te = comp.load_train(), comp.load_test()
    if comp.THIN:  # the half-density copy of train: October has half the ordinary payments per day (make_thin.py)
        drop = np.load('preds/thin_rows.npy')
        if comp.THIN == '_thinsep':  # only September thinned: stands in for "train at full density, test at half"
            drop = drop[comp.fixed_folds(tr.iloc[drop]) == 2]
        tr = tr.drop(index=drop)
    d = pd.concat([tr.drop(columns=[comp.TARGET]), te], ignore_index=True)
    d['t'] = pd.to_datetime(d['timestamp'])
    return d


def codes(s):
    return pd.Categorical(s.fillna('none')).codes.astype(np.int16)


def block_row(d):
    """What one row says on its own, plus account / merchant details and whose phone it is."""
    ac = comp.load_accounts().set_index('account_id')
    mr = comp.load_merchants().set_index('merchant_id')
    X = pd.DataFrame(index=d.index)
    X['hour'] = d.t.dt.hour + d.t.dt.minute / 60
    X['weekday'] = d.t.dt.weekday
    for c in ['amount', 'sim_age_hours', 'device_age_days', 'sender_account_age_days', 'new_payee', 'pin_retries',
              'login_to_txn_sec', 'balance_before', 'qr_vpa_mismatch']:
        X[c] = d[c]
    a = d.amount
    X['amount_share_of_balance'] = a / d.balance_before
    X['balance_after'] = d.balance_before - a
    X['amount_is_whole'] = (a == a.round()).astype(np.int8)
    for m in (10, 100, 500, 1000):
        X[f'amount_mod_{m}'] = a % m
    for m in (1000, 5000, 10000):   # how far under the next round limit (1,999 / 4,999 / ...)
        X[f'under_{m}'] = (m - a % m) % m
    X['amount_paise'] = ((a * 100).round() % 100)
    for c in ['receiver_type', 'txn_type', 'merchant_category', 'app', 'device_city']:
        X[c] = codes(d[c])
    # whose phone: an account's own phone carries its number (AC377784 <-> DV0377784); 420000+ are extra phones
    dev, snd = d.device_id.str[3:].astype(int), d.sender_id.str[2:].astype(int)
    X['phone_not_own'] = (dev != snd).astype(np.int8)
    X['phone_extra'] = (dev >= len(ac)).astype(np.int8)
    # sender's account
    s = ac.reindex(d.sender_id.values)
    X['away_from_home'] = (d.device_city.values != s.home_city.values).astype(np.int8)
    X['s_kyc'], X['s_os'], X['s_bank'] = codes(s.kyc_level).copy(), codes(s.os).copy(), codes(s.bank_code).copy()
    X['s_home_city'] = codes(pd.Series(s.home_city.values))
    # receiver: a person's account or a merchant
    r = ac.reindex(d.receiver_id.values)
    X['r_account_age_days'] = (d.t.values - pd.to_datetime(r.opened_on).values) / np.timedelta64(1, 'D')
    X['r_kyc'], X['r_bank'] = codes(pd.Series(r.kyc_level.values)), codes(pd.Series(r.bank_code.values))
    X['r_same_home_city'] = (r.home_city.values == s.home_city.values).astype(np.int8)
    X['r_same_bank'] = (r.bank_code.values == s.bank_code.values).astype(np.int8)
    m = mr.reindex(d.receiver_id.values)
    X['m_age_days'] = (d.t.values - pd.to_datetime(m.onboarded_on).values) / np.timedelta64(1, 'D')
    X['m_same_city_as_phone'] = (m.city.values == d.device_city.values).astype(np.int8)
    X['m_category_differs'] = ((m.category.values != d.merchant_category.values) & m.category.notna().values).astype(np.int8)
    X['m_city'] = codes(pd.Series(m.city.values))
    return X


BIG = 1 << 25   # more than the data's time span in seconds plus the widest window, so keys never overlap
# "All-time" windows. The data spans 131.6 days, so BIG // 4 (97 days) is a little short; ALL_DAYS=140 covers it all.
# sub1-sub26 were built with the default, so it stays the default for reproduce.sh.
import os as _os
ALL = int(_os.environ['ALL_DAYS']) * 86400 if _os.environ.get('ALL_DAYS') else BIG // 4
M, H, D = 60, 3600, 86400


def ids(d):
    """Integer codes: accounts keep their number, merchants get 1,000,000 + number, phones their number."""
    snd = d.sender_id.str[2:].astype(np.int64).values
    rcv = d.receiver_id.str[2:].astype(np.int64).values + np.where(d.receiver_id.str[:2].values == 'MR', 1_000_000, 0)
    dev = d.device_id.str[3:].astype(np.int64).values
    ts = ((d.t - d.t.min()) / np.timedelta64(1, 's')).astype(np.int64).values
    return snd, rcv, dev, ts


class Events:
    """Rows keyed by an entity and sorted by (key, time), for window counts / sums and nearest-event lookups."""

    def __init__(self, key, ts):
        c = key * BIG + ts
        self.o = np.argsort(c, kind='stable')
        self.c = c[self.o]

    def win(self, qkey, qts, lo, hi, val=None):
        """Per query (key, time): count, or sum of val, over events of that key with time in [t+lo, t+hi]."""
        q = qkey * BIG + qts
        a, b = np.searchsorted(self.c, q + lo, 'left'), np.searchsorted(self.c, q + hi, 'right')
        if val is None:
            return (b - a).astype(np.float32)
        cs = np.r_[0.0, np.cumsum(np.asarray(val, np.float64)[self.o])]
        return (cs[b] - cs[a]).astype(np.float32)

    def nearest(self, qkey, qts, before=True):
        """Row number of the event of that key just before (or after) time t, -1 when there is none."""
        q = qkey * BIG + qts
        i = np.searchsorted(self.c, q, 'left') - 1 if before else np.searchsorted(self.c, q, 'right')
        ok = (i >= 0) & (i < len(self.c))
        j = np.clip(i, 0, len(self.c) - 1)
        ok &= self.c[j] // BIG == qkey
        return np.where(ok, self.o[j], -1)


def first_in(key, ts, gap):
    """1 where a row is the first of its key for `gap` seconds (so a window sum counts distinct keys, roughly)."""
    o = np.lexsort((ts, key))
    k, t = key[o], ts[o]
    f = np.ones(len(key), np.float32)
    f[o[1:]] = ((k[1:] != k[:-1]) | (t[1:] - t[:-1] > gap)).astype(np.float32)
    return f


def gap(E, key, ts, before):
    """Seconds to the previous (or next) event of that key, NaN when there is none."""
    j = E.nearest(key, ts, before)
    return np.where(j >= 0, np.abs(ts - ts[j]), np.nan).astype(np.float32)


def block_conn(d):
    """How a row connects to other rows: bursts at the receiver, at the phone and from the sender; money moving on;
    payments in the opposite direction; refund runs; round-number sales. Windows look both back and ahead in time."""
    snd, rcv, dev, ts = ids(d)
    amt = d.amount.values.astype(np.float64)
    n = len(d)
    X = pd.DataFrame(index=d.index)
    pair = snd * 2_000_000 + rcv                       # sender -> receiver
    rev = rcv * 2_000_000 + snd                        # the same two, the other way round
    _, pair_c = np.unique(pair, return_inverse=True)
    S, R, P = Events(snd, ts), Events(rcv, ts), Events(dev, ts)
    is_p2p = d.receiver_type.values == 'P2P'
    typ = d.txn_type.values
    round500 = (amt % 500 == 0).astype(np.float64)
    whole = (amt == np.round(amt)).astype(np.float64)
    mism = d.qr_vpa_mismatch.fillna(0).values

    # --- the receiver: who else pays it, how much, how fast
    for name, w in (('10m', 10 * M), ('1h', H), ('6h', 6 * H), ('24h', D)):
        X[f'r_in_n_{name}'] = R.win(rcv, ts, -w, w)
        X[f'r_in_sum_{name}'] = R.win(rcv, ts, -w, w, amt)
    X['r_in_n_all'] = R.win(rcv, ts, -ALL, ALL)
    for name, w in (('1h', H), ('24h', D)):
        X[f'r_senders_{name}'] = R.win(rcv, ts, -w, w, first_in(pair, ts, w))
    X['r_senders_all'] = R.win(rcv, ts, -ALL, ALL, first_in(pair, ts, BIG))
    X['r_new_payee_share_24h'] = R.win(rcv, ts, -D, D, d.new_payee.values) / X.r_in_n_24h
    X['r_qr_share_24h'] = R.win(rcv, ts, -D, D, typ == 'QR') / X.r_in_n_24h
    X['r_collect_share_24h'] = R.win(rcv, ts, -D, D, typ == 'COLLECT') / X.r_in_n_24h
    X['r_mismatch_24h'] = R.win(rcv, ts, -D, D, mism)
    X['r_mismatch_all'] = R.win(rcv, ts, -ALL, ALL, mism)
    X['r_round_share_24h'] = R.win(rcv, ts, -D, D, round500) / X.r_in_n_24h
    X['r_round_n_24h'] = R.win(rcv, ts, -D, D, round500)
    X['r_whole_share_24h'] = R.win(rcv, ts, -D, D, whole) / X.r_in_n_24h
    X['r_amount_vs_mean_24h'] = amt / (X.r_in_sum_24h / X.r_in_n_24h)
    X['r_gap_prev'], X['r_gap_next'] = gap(R, rcv, ts, True), gap(R, rcv, ts, False)
    # money leaving the receiver again (the receiver as a sender)
    for name, w in (('10m', 10 * M), ('1h', H), ('6h', 6 * H)):
        X[f'r_out_n_{name}'] = S.win(rcv, ts, 1, w)
        X[f'r_out_sum_{name}'] = S.win(rcv, ts, 1, w, amt)
    X['r_out_n_all'] = S.win(rcv, ts, -ALL, ALL)
    X['r_out_over_in_1h'] = X.r_out_sum_1h / X.r_in_sum_1h
    j = S.nearest(rcv, ts, False)                       # the receiver's next payment
    X['r_next_out_gap'] = np.where(j >= 0, ts[j] - ts, np.nan)
    X['r_next_out_ratio'] = np.where(j >= 0, amt[j] / amt, np.nan)
    nxt = np.where((j >= 0) & (ts[j] - ts <= H) & (amt[j] / amt >= 0.7) & (amt[j] / amt <= 1.02) & is_p2p, j, -1)

    # --- the sender: bursts, and money that came in just before
    for name, w in (('1m', M), ('10m', 10 * M), ('1h', H), ('24h', D)):
        X[f's_out_n_{name}'] = S.win(snd, ts, -w, w)
        X[f's_out_sum_{name}'] = S.win(snd, ts, -w, w, amt)
    X['s_out_n_all'] = S.win(snd, ts, -ALL, ALL)
    for name, w in (('1h', H), ('24h', D)):
        X[f's_receivers_{name}'] = S.win(snd, ts, -w, w, first_in(pair, ts, w))
    X['s_receivers_all'] = S.win(snd, ts, -ALL, ALL, first_in(pair, ts, BIG))
    X['s_drain_1h'] = S.win(snd, ts, 0, H, amt) / d.balance_before.values
    X['s_gap_prev'], X['s_gap_next'] = gap(S, snd, ts, True), gap(S, snd, ts, False)
    jp = S.nearest(snd, ts)
    city = codes(d.device_city).astype(np.int64)
    X['s_city_changed'] = ((jp >= 0) & (city[jp] != city)).astype(np.int8)
    X['s_phone_changed'] = ((jp >= 0) & (dev[jp] != dev)).astype(np.int8)
    X['s_amount_vs_median'] = amt / d.groupby('sender_id').amount.transform('median').values
    for name, w in (('7d', 7 * D), ('30d', 30 * D)):
        X[f's_refunds_{name}'] = S.win(snd, ts, -w, w, typ == 'REFUND')
        X[f's_lending_{name}'] = S.win(snd, ts, -w, w, d.merchant_category.values == 'lending')
    X['s_refunds_all'] = S.win(snd, ts, -ALL, ALL, typ == 'REFUND')
    X['s_collects_24h'] = S.win(snd, ts, -D, D, typ == 'COLLECT')
    X['s_new_payee_share_24h'] = S.win(snd, ts, -D, D, d.new_payee.values) / X.s_out_n_24h
    X['s_cities_24h'] = S.win(snd, ts, -D, D, first_in(snd * 1000 + city, ts, D))
    for name, w in (('1h', H), ('24h', D)):               # the sender as a receiver, before this payment
        X[f's_in_n_{name}'] = R.win(snd, ts, -w, -1)
        X[f's_in_sum_{name}'] = R.win(snd, ts, -w, -1, amt)
    X['s_in_n_all'] = R.win(snd, ts, -ALL, ALL)
    X['s_out_over_in_1h'] = amt / X.s_in_sum_1h
    j = R.nearest(snd, ts)                              # the last payment the sender received
    X['s_last_in_gap'] = np.where(j >= 0, ts - ts[j], np.nan)
    X['s_last_in_ratio'] = np.where(j >= 0, amt / amt[j], np.nan)
    prv = np.where((j >= 0) & (ts - ts[j] <= H) & (amt / amt[j] >= 0.7) & (amt / amt[j] <= 1.02), j, -1)

    # --- chains: hops linked when most of the money moves on within the hour
    back, fwd = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for _ in range(12):
        back = np.where(prv >= 0, 1 + back[prv], 0).astype(np.float32)
        fwd = np.where(nxt >= 0, 1 + fwd[nxt], 0).astype(np.float32)
    X['chain_back'], X['chain_fwd'], X['chain_len'] = back, fwd, back + fwd + 1

    # --- the phone
    for name, w in (('1m', M), ('10m', 10 * M), ('1h', H), ('24h', D)):
        X[f'p_n_{name}'] = P.win(dev, ts, -w, w)
    X['p_n_all'] = P.win(dev, ts, -ALL, ALL)
    X['p_sum_10m'] = P.win(dev, ts, -10 * M, 10 * M, amt)
    sd = snd * 1_000_000 + dev
    X['p_senders_all'] = P.win(dev, ts, -ALL, ALL, first_in(sd, ts, BIG))
    X['p_senders_24h'] = P.win(dev, ts, -D, D, first_in(sd, ts, D))
    X['p_receivers_10m'] = P.win(dev, ts, -10 * M, 10 * M, first_in(dev * 2_000_000 + rcv, ts, 10 * M))
    X['s_phones_all'] = S.win(snd, ts, -ALL, ALL, first_in(sd, ts, BIG))
    X['sp_n_all'] = pd.Series(ts).groupby(sd).transform('size').values.astype(np.float32)   # this account on this phone
    X['sp_since_first'] = ts - pd.Series(ts).groupby(sd).transform('min').values
    X['sp_share_of_sender'] = X.sp_n_all / X.s_out_n_all
    X['p_since_first'] = ts - pd.Series(ts).groupby(dev).transform('min').values
    X['p_gap_prev'], X['p_gap_next'] = gap(P, dev, ts, True), gap(P, dev, ts, False)

    # --- this sender and this receiver
    _, rev_c = np.unique(np.r_[pair, rev], return_inverse=True)   # pairs and reversed pairs on one code scale
    pc, rc = rev_c[:n].astype(np.int64), rev_c[n:].astype(np.int64)
    PR = Events(pc, ts)
    for name, w in (('1h', H), ('24h', D), ('7d', 7 * D)):
        X[f'pair_n_{name}'] = PR.win(pc, ts, -w, w)
    X['pair_n_all'] = PR.win(pc, ts, -ALL, ALL)
    X['pair_sum_24h'] = PR.win(pc, ts, -D, D, amt)
    X['pair_round_24h'] = PR.win(pc, ts, -D, D, round500)
    X['pair_refunds_30d'] = PR.win(pc, ts, -30 * D, 30 * D, typ == 'REFUND')
    X['pair_since_first'] = ts - pd.Series(ts).groupby(pair_c).transform('min').values
    X['pair_share_of_receiver_24h'] = X.pair_n_24h / X.r_in_n_24h
    X['rev_n_all'] = PR.win(rc, ts, -ALL, ALL)                 # payments the other way round
    X['rev_n_24h'] = PR.win(rc, ts, -D, D)
    j = PR.nearest(rc, ts)
    X['rev_last_gap'] = np.where(j >= 0, ts - ts[j], np.nan)
    X['rev_last_ratio'] = np.where(j >= 0, amt / amt[j], np.nan)
    X['rev_last_amount'] = np.where(j >= 0, amt[j], np.nan)
    j = PR.nearest(rc, ts, False)
    X['rev_next_gap'] = np.where(j >= 0, ts[j] - ts, np.nan)
    X['rev_next_ratio'] = np.where(j >= 0, amt[j] / amt, np.nan)
    return X.replace([np.inf, -np.inf], np.nan)


def sessions(key, ts, gap):
    """Session number per row: rows of one key belong together while the gaps between them stay within `gap` seconds."""
    o = np.lexsort((ts, key))
    k, t = key[o], ts[o]
    new = np.r_[True, (k[1:] != k[:-1]) | (t[1:] - t[:-1] > gap)]
    sid = np.empty(len(key), np.int64)
    sid[o] = np.cumsum(new) - 1
    return sid


def block_ring(d):
    """The ring-level view: every row gets statistics of the event it belongs to (the sender's session, the receiver's
    session, the phone's burst, this account on this phone, this sender-receiver pair). Per-row columns such as
    sim_age_hours are noisy; over an event they are not."""
    snd, rcv, dev, ts = ids(d)
    ac = comp.load_accounts().set_index('account_id')
    amt = d.amount.values.astype(np.float64)
    hour = d.t.dt.hour.values
    typ = d.txn_type.values
    V = pd.DataFrame({
        'amount': amt, 'sim_age': d.sim_age_hours.values, 'dev_age': d.device_age_days.values,
        'login': d.login_to_txn_sec.values.astype(np.float64), 'pin': d.pin_retries.values.astype(np.float64),
        'new_payee': d.new_payee.values.astype(np.float64), 'balance': d.balance_before.values,
        'away': (d.device_city.values != ac.home_city.reindex(d.sender_id.values).values).astype(np.float64),
        'whole': (amt == np.round(amt)).astype(np.float64), 'round500': (amt % 500 == 0).astype(np.float64),
        'night': ((hour >= 22) | (hour < 5)).astype(np.float64), 'evening': ((hour >= 19) & (hour < 24)).astype(np.float64),
        'p2p': (d.receiver_type.values == 'P2P').astype(np.float64),
        'collect': (typ == 'COLLECT').astype(np.float64), 'qr': (typ == 'QR').astype(np.float64),
        'refund': (typ == 'REFUND').astype(np.float64), 'mismatch': d.qr_vpa_mismatch.fillna(0).values,
        'extra': (dev >= len(ac)).astype(np.float64), 'not_own': (dev != snd).astype(np.float64),
        'sim_fresh': (d.sim_age_hours.values < 72).astype(np.float64), 'dev_fresh': (d.device_age_days.values < 2).astype(np.float64),
        'lending': (d.merchant_category.values == 'lending').astype(np.float64),
        'ts': ts.astype(np.float64), 'snd': snd, 'rcv': rcv, 'dev': dev})
    out = {}

    def add(prefix, sid, mean=(), mn=(), mx=(), sd=(), uniq=()):
        g = V.groupby(sid, sort=False)
        n = g['ts'].transform('size').values.astype(np.float32)
        t0, t1 = g['ts'].transform('min').values, g['ts'].transform('max').values
        out[f'{prefix}_n'], out[f'{prefix}_dur'] = n, (t1 - t0).astype(np.float32)
        out[f'{prefix}_pos'] = ((ts - t0) / np.maximum(t1 - t0, 1)).astype(np.float32)
        for how, cols in (('mean', mean), ('min', mn), ('max', mx), ('std', sd), ('nunique', uniq)):
            if cols:
                T = g[list(cols)].transform(how)
                for c in cols:
                    out[f'{prefix}_{c}_{how}'] = T[c].values.astype(np.float32)
        return g

    # the sender's session (gaps up to 2 hours)
    sid = sessions(snd, ts, 2 * H)
    g = add('ss', sid, mean=('away', 'new_payee', 'pin', 'login', 'whole', 'night', 'evening', 'p2p', 'collect', 'extra',
                              'not_own', 'sim_fresh', 'dev_fresh', 'amount', 'lending', 'refund'),
            mn=('sim_age', 'dev_age', 'login'), mx=('pin', 'amount', 'balance'), sd=('amount',), uniq=('rcv', 'dev'))
    out['ss_amount_sum_over_balance'] = (g['amount'].transform('sum').values / out['ss_balance_max']).astype(np.float32)
    # the receiver's session (gaps up to 6 hours)
    sid = sessions(rcv, ts, 6 * H)
    add('rs', sid, mean=('amount', 'new_payee', 'whole', 'round500', 'sim_fresh', 'dev_fresh', 'away', 'night', 'evening',
                         'qr', 'mismatch', 'collect', 'extra', 'not_own', 'login', 'pin'),
        mn=('amount', 'sim_age'), mx=('amount',), sd=('amount',), uniq=('snd',))
    out['rs_amount_cv'] = out['rs_amount_std'] / out['rs_amount_mean']
    # the phone's burst (gaps up to 10 minutes)
    sid = sessions(dev, ts, 10 * M)
    add('ps', sid, mean=('whole', 'login', 'amount', 'new_payee'), sd=('amount',), uniq=('snd', 'rcv'))
    # this account on this phone, all time
    add('sp', snd * 1_000_000 + dev, mean=('away', 'night', 'new_payee', 'whole', 'sim_fresh', 'p2p'),
        mn=('sim_age', 'dev_age'), mx=('pin',))
    # this sender and this receiver, all time
    add('pr', snd * 2_000_000 + rcv, mean=('amount', 'round500', 'whole', 'refund', 'collect', 'qr', 'evening', 'mismatch'),
        sd=('amount',), mx=('amount',))
    X = pd.DataFrame(out, index=d.index)
    return X.replace([np.inf, -np.inf], np.nan)


def block_flag(d):
    """The QR mismatch flag as explicit states, and how many "marked" rows share an event with the row.
    In train an ordinary row carries a flag exactly when it is a QR payment, and never pays a person by QR. A row that
    breaks either rule is fraud almost every time: 87.8% of them carry a fraud label, which is what 13% unreported fraud
    leaves. October changes the raw column (44% of its QR-to-person rows have a blank flag, a combination train never
    shows, and models trained on the raw column call most of them legit), so models get these states instead of the raw
    column (train_gbm.py --drop qr_vpa_mismatch)."""
    snd, rcv, dev, ts = ids(d)
    qr, has = d.txn_type.values == 'QR', d.qr_vpa_mismatch.notna().values
    p2p = d.receiver_type.values == 'P2P'
    odd = qr != has
    X = pd.DataFrame(index=d.index)
    X['flag_one'] = (d.qr_vpa_mismatch.values == 1).astype(np.int8)
    X['flag_odd'] = odd.astype(np.int8)                      # a value without a QR payment, or a QR payment without one
    X['flag_value_not_qr'] = (~qr & has).astype(np.int8)
    X['flag_blank_qr'] = (qr & ~has).astype(np.int8)
    X['qr_to_person'] = (qr & p2p).astype(np.int8)
    X['person_with_category'] = (p2p & d.merchant_category.notna().values).astype(np.int8)
    mark = (odd | (qr & p2p)).astype(np.float64)             # fraud on the row's own evidence
    X['marked'] = mark.astype(np.int8)
    S, R, P = Events(snd, ts), Events(rcv, ts), Events(dev, ts)
    # other marked rows in the same event: at the receiver, from the sender, on the phone, between the two
    X['r_marked_24h'] = R.win(rcv, ts, -D, D, mark) - mark
    X['r_marked_all'] = R.win(rcv, ts, -ALL, ALL, mark) - mark
    X['r_marked_share_24h'] = R.win(rcv, ts, -D, D, mark) / R.win(rcv, ts, -D, D)
    X['r_marked_share_all'] = R.win(rcv, ts, -ALL, ALL, mark) / R.win(rcv, ts, -ALL, ALL)
    X['s_marked_1h'] = S.win(snd, ts, -H, H, mark) - mark
    X['s_marked_24h'] = S.win(snd, ts, -D, D, mark) - mark
    X['s_marked_all'] = S.win(snd, ts, -ALL, ALL, mark) - mark
    X['s_marked_share_24h'] = S.win(snd, ts, -D, D, mark) / S.win(snd, ts, -D, D)
    X['p_marked_10m'] = P.win(dev, ts, -10 * M, 10 * M, mark) - mark
    X['p_marked_all'] = P.win(dev, ts, -ALL, ALL, mark) - mark
    X['p_marked_share_10m'] = P.win(dev, ts, -10 * M, 10 * M, mark) / P.win(dev, ts, -10 * M, 10 * M)
    X['pair_marked_all'] = pd.Series(mark).groupby(snd * 2_000_000 + rcv).transform('sum').values - mark
    # along the money: marked payments into the sender before, and out of the receiver after
    X['s_in_marked_24h'] = R.win(snd, ts, -D, -1, mark)
    X['r_out_marked_24h'] = S.win(rcv, ts, 1, D, mark)
    X['r_out_marked_all'] = S.win(rcv, ts, -ALL, ALL, mark)
    X['s_in_marked_all'] = R.win(snd, ts, -ALL, ALL, mark)
    return X.replace([np.inf, -np.inf], np.nan)


def block_extra(d):
    """Gaps found in review (14 Nov): refunds seen from the merchant and the phone, longer windows for phone diversity,
    same-direction repeats between a pair (a purchase followed by a refund of the same amount), the regularity of an
    entity's payment gaps, phones per merchant, and the merchant-category mismatch spread over the event."""
    snd, rcv, dev, ts = ids(d)
    amt = d.amount.values.astype(np.float64)
    typ = d.txn_type.values
    refund = (typ == 'REFUND').astype(np.float64)
    S, R, P = Events(snd, ts), Events(rcv, ts), Events(dev, ts)
    X = pd.DataFrame(index=d.index)
    for name, w in (('24h', D), ('7d', 7 * D), ('all', ALL)):
        n = R.win(rcv, ts, -w, w)
        X[f'r_refund_n_{name}'] = R.win(rcv, ts, -w, w, refund)
        X[f'r_refund_share_{name}'] = X[f'r_refund_n_{name}'] / n
        X[f'p_refund_n_{name}'] = P.win(dev, ts, -w, w, refund)
        X[f'r_refund_senders_{name}'] = R.win(rcv, ts, -w, w, refund * first_in(snd * 2_000_000 + rcv, ts, w if w < ALL else BIG))
    for name, w in (('1h', H), ('24h', D), ('all', ALL)):
        X[f'p_receivers_{name}'] = P.win(dev, ts, -w, w, first_in(dev * 2_000_000 + rcv, ts, w if w < ALL else BIG))
    for name, gap in (('30m', 30 * M), ('2h', 2 * H)):
        sid = sessions(dev, ts, gap)
        g = pd.DataFrame({'sid': sid, 'rcv': rcv, 'snd': snd, 'login': d.login_to_txn_sec.values, 'ts': ts}).groupby('sid')
        X[f'ps{name}_n'] = g.ts.transform('size').values
        X[f'ps{name}_rcv_nunique'] = g.rcv.transform('nunique').values
        X[f'ps{name}_snd_nunique'] = g.snd.transform('nunique').values
        X[f'ps{name}_login_mean'] = g.login.transform('mean').values
    # a pair's previous and next payment in the same direction (purchase -> refund, repeated fake sales)
    _, pc = np.unique(snd * 2_000_000 + rcv, return_inverse=True)
    PC = Events(pc.astype(np.int64), ts)
    for nm, before in (('last', True), ('next', False)):
        j = PC.nearest(pc.astype(np.int64), ts, before)
        X[f'pair_{nm}_gap'] = np.where(j >= 0, np.abs(ts - ts[j]), np.nan)
        X[f'pair_{nm}_ratio'] = np.where(j >= 0, amt / amt[np.maximum(j, 0)], np.nan)
        X[f'pair_{nm}_type_refund'] = np.where(j >= 0, refund[np.maximum(j, 0)], np.nan)
    # regularity of gaps (scripted payments come at steady intervals)
    for nm, key in (('s', snd), ('p', dev)):
        o = np.lexsort((ts, key)); k, t = key[o], ts[o]
        gp = np.r_[np.nan, np.where(k[1:] == k[:-1], t[1:] - t[:-1], np.nan)].astype(np.float64)
        gser = pd.Series(gp)
        grp = pd.Series(sessions(key, ts, 2 * H)[o])
        stats = gser.groupby(grp).agg(['mean', 'std', 'min'])
        out = stats.reindex(grp.values).values
        back = np.empty_like(out); back[o] = out
        X[f'{nm}_gap_mean_2h'], X[f'{nm}_gap_std_2h'], X[f'{nm}_gap_min_2h'] = back[:, 0], back[:, 1], back[:, 2]
        X[f'{nm}_gap_cv_2h'] = back[:, 1] / back[:, 0]
    # phones per merchant and per receiver session
    X['r_phones_24h'] = R.win(rcv, ts, -D, D, first_in(rcv * 1_000_000 + dev, ts, D))
    X['r_phones_per_sender_24h'] = X.r_phones_24h / R.win(rcv, ts, -D, D, first_in(snd * 2_000_000 + rcv, ts, D))
    # merchant-category mismatches and QR-to-person rows spread over the receiver, sender and phone
    ac = comp.load_merchants().set_index('merchant_id')
    mcat = ac.category.reindex(d.receiver_id.values).values
    odd = ((pd.notna(mcat) & (mcat != d.merchant_category.values)) | ((d.receiver_type.values == 'P2P') & d.merchant_category.notna().values)).astype(np.float64)
    X['cat_odd'] = odd
    X['r_cat_odd_all'] = R.win(rcv, ts, -ALL, ALL, odd) - odd
    X['s_cat_odd_all'] = S.win(snd, ts, -ALL, ALL, odd) - odd
    X['p_cat_odd_all'] = P.win(dev, ts, -ALL, ALL, odd) - odd
    return X.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def block_ip(d):
    """The IP-reputation score, which the base model leaves out: raw, and as a rank within its own month (the vendor
    rescaled it in October, so the rank is the part that might carry over)."""
    month = d.t.dt.to_period('M').astype(str).values
    X = pd.DataFrame(index=d.index)
    X['ip_raw'] = d.ip_risk_score.values.astype(np.float32)
    X['ip_rank_month'] = d.ip_risk_score.groupby(month).rank(pct=True).values.astype(np.float32)
    snd, rcv, dev, ts = ids(d)
    X['ip_rank_sender_mean'] = pd.Series(X.ip_rank_month.values).groupby(snd).transform('mean').values.astype(np.float32)
    X['ip_rank_receiver_mean'] = pd.Series(X.ip_rank_month.values).groupby(rcv).transform('mean').values.astype(np.float32)
    return X


def block_graph(d):
    """Two-hop and component structure. Per day (and per 3-day bucket): accounts are linked by person-to-person payments
    and by sharing a phone; every row gets the size and makeup of the component it belongs to. Two-hop: how many other
    people a row's receiver's senders paid that day, and how many other senders a row's sender's receivers had.
    Merchant side: is the merchant in the sender's home city, and how local a merchant's customers are."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    snd, rcv, dev, ts = ids(d)
    n = len(d)
    p2p = d.receiver_type.values == 'P2P'
    ac = comp.load_accounts().set_index('account_id')
    mr = comp.load_merchants().set_index('merchant_id')
    X = pd.DataFrame(index=d.index)
    for name, span in (('1d', D), ('3d', 3 * D)):
        b = ts // span
        # nodes: (account, bucket) and (phone, bucket); edges: sender-receiver for P2P rows, sender-phone for every row
        a_s = pd.factorize(pd.Series(snd) * 100000 + b)[0]
        node_acc = pd.Series(np.r_[snd, rcv[p2p]] * 100000 + np.r_[b, b[p2p]])
        acc_codes, acc_uni = pd.factorize(node_acc)
        s_node = acc_codes[:n]; r_node = acc_codes[n:]
        dv_codes = pd.factorize(pd.Series(dev) * 100000 + b)[0] + len(acc_uni)
        N = len(acc_uni) + dv_codes.max() + 1
        rows = np.r_[s_node[p2p], s_node]; cols = np.r_[r_node, dv_codes]
        g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(N, N))
        _, lab = connected_components(g, directed=False)
        comp_id = lab[s_node]
        cs = pd.DataFrame({'c': comp_id, 'snd': snd, 'rcv': rcv, 'dev': dev, 'amt': d.amount.values, 'p2p': p2p,
                           'new': d.new_payee.values, 'whole': (d.amount.values == np.round(d.amount.values))})
        gr = cs.groupby('c')
        X[f'g{name}_rows'] = gr.snd.transform('size').values
        X[f'g{name}_accounts'] = gr.snd.transform('nunique').values
        X[f'g{name}_receivers'] = gr.rcv.transform('nunique').values
        X[f'g{name}_phones'] = gr.dev.transform('nunique').values
        X[f'g{name}_p2p_share'] = gr.p2p.transform('mean').values
        X[f'g{name}_new_share'] = gr.new.transform('mean').values
        X[f'g{name}_whole_share'] = gr.whole.transform('mean').values
        X[f'g{name}_amt_sum'] = gr.amt.transform('sum').values
        X[f'g{name}_phones_per_acc'] = X[f'g{name}_phones'] / X[f'g{name}_accounts']
    # two-hop, per day
    day = ts // D
    t = pd.DataFrame({'s': snd, 'r': rcv, 'day': day})
    s_deg = t.groupby(['s', 'day']).r.transform('nunique').values          # receivers this sender paid that day
    r_deg = t.groupby(['r', 'day']).s.transform('nunique').values          # senders this receiver had that day
    t['s_deg'], t['r_deg'] = s_deg, r_deg
    X['h2_rcv_senders_outdeg_mean'] = t.groupby(['r', 'day']).s_deg.transform('mean').values
    X['h2_rcv_senders_outdeg_max'] = t.groupby(['r', 'day']).s_deg.transform('max').values
    X['h2_snd_receivers_indeg_mean'] = t.groupby(['s', 'day']).r_deg.transform('mean').values
    X['h2_snd_receivers_indeg_max'] = t.groupby(['s', 'day']).r_deg.transform('max').values
    # merchant side
    home = ac.home_city.reindex(d.sender_id.values).values
    mcity = mr.city.reindex(d.receiver_id.values).values
    X['m_in_home_city'] = np.where(pd.isna(mcity), np.nan, (mcity == home).astype(float))
    loc = pd.Series(np.where(pd.isna(mcity), np.nan, (mcity == d.device_city.values).astype(float)))
    X['m_local_share'] = loc.groupby(d.receiver_id.values).transform('mean').values
    X['m_home_share'] = pd.Series(X.m_in_home_city.values).groupby(d.receiver_id.values).transform('mean').values
    return X.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def block_svd(d):
    """Global graph position: truncated SVD of the whole sender x receiver payment graph and of the account x phone
    graph (train + test, all months, no labels). Rows get 12-dimensional sender and receiver embeddings and how close
    the two are; accounts in the same community land near each other even when they never paid each other."""
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    s_code, s_uni = pd.factorize(d.sender_id)
    r_code, r_uni = pd.factorize(d.receiver_id)
    dv_code, dv_uni = pd.factorize(d.device_id)
    A = coo_matrix((np.log1p(np.ones(len(d))), (s_code, r_code)), shape=(len(s_uni), len(r_uni))).tocsr()
    A.data = np.log1p(A.data)
    svd = TruncatedSVD(12, random_state=0).fit(A)
    S_emb = normalize(svd.transform(A))                                  # senders
    R_emb = normalize(svd.components_.T * svd.singular_values_)          # receivers
    B = coo_matrix((np.ones(len(d)), (s_code, dv_code)), shape=(len(s_uni), len(dv_uni))).tocsr()
    B.data = np.log1p(B.data)
    svd2 = TruncatedSVD(8, random_state=0).fit(B)
    SD = normalize(svd2.transform(B))
    X = pd.DataFrame(index=d.index)
    se, re_ = S_emb[s_code], R_emb[r_code]
    for k in range(12):
        X[f'svd_s{k}'] = se[:, k]
        X[f'svd_r{k}'] = re_[:, k]
    X['svd_sr_cos'] = (se * re_).sum(1)
    for k in range(8):
        X[f'svd_sd{k}'] = SD[s_code][:, k]
    return X.astype(np.float32)


def block_spike(d):
    """How unusual this month is for the account / phone / merchant: its rows, partners and new payees this month
    against its average over the other months, with every month scaled to the same total volume (October has half the
    rows). An account that is quiet for three months and busy in one stands out, whichever month that is."""
    month = np.minimum(d.t.dt.month.values, 10)                         # early-November rows are October's tail
    tot = pd.Series(month).value_counts()
    scale = (tot.mean() / tot).reindex(month).values                    # volume per month, equalised
    X = pd.DataFrame(index=d.index)
    for nm, key in (('s', d.sender_id.values), ('r', d.receiver_id.values), ('p', d.device_id.values)):
        f = pd.DataFrame({'k': key, 'm': month, 'w': scale, 'other': d.receiver_id.values if nm != 'r' else d.sender_id.values,
                          'new': d.new_payee.values * scale})
        per = f.groupby(['k', 'm']).agg(n=('w', 'sum'), uniq=('other', 'nunique'), new=('new', 'sum'))
        allk = per.groupby(level=0).agg(n=('n', 'sum'), uniq=('uniq', 'sum'), new=('new', 'sum'), months=('n', 'size'))
        here = per.reindex(pd.MultiIndex.from_arrays([key, month])).values
        tot_k = allk.reindex(key)
        n_other_months = max(len(tot) - 1, 1)
        other = (tot_k[['n', 'uniq', 'new']].values - here) / n_other_months
        X[f'{nm}_month_n'] = here[:, 0]
        X[f'{nm}_spike_n'] = here[:, 0] / (other[:, 0] + 0.5)
        X[f'{nm}_spike_uniq'] = here[:, 1] / (other[:, 1] + 0.5)
        X[f'{nm}_spike_new'] = here[:, 2] / (other[:, 2] + 0.5)
        X[f'{nm}_months_active'] = tot_k['months'].values
    return X.replace([np.inf, -np.inf], np.nan).astype(np.float32)


BLOCKS = {'row': block_row, 'conn': block_conn, 'ring': block_ring, 'flag': block_flag, 'extra': block_extra, 'ip': block_ip,
          'graph': block_graph, 'svd': block_svd, 'spike': block_spike}

if __name__ == '__main__':
    names = sys.argv[1:] or list(BLOCKS)
    d = both()
    for n in names:
        t0 = time.time()
        X = BLOCKS[n](d)
        X.index = d[comp.ID].values
        X = X.astype({c: np.float32 for c in X.columns if X[c].dtype == np.float64})
        X.to_parquet(f'data/feat_{n}{comp.THIN}.parquet')
        print(f'feat_{n}{comp.THIN}: {X.shape[1]} features, {len(X)} rows, {time.time() - t0:.0f}s', flush=True)
