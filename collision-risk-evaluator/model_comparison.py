"""Compare every model family on the official target: each event's final tier from messages >= 2 days out.

Run after final_tier_pipeline.py inputs are in data/:
    python model_comparison.py

Uses development events only (5-fold by event); the held-out events from final_tier_pipeline.py are never touched.
Each model is scored two ways:
  alone         its own three-class call, with a Critical threshold
  with forecast the distance forecast sets the tier, and the model only escalates to Critical
Thresholds are cross-fitted: chosen on the other four folds, then applied to the fold being scored.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from final_tier_pipeline import LABELS, OUT, RECOMMENDED, SEED, TIERS, decide, distance_tier, event_features, oof_predict, score

warnings.filterwarnings('ignore')
CLS = {t: i for i, t in enumerate(TIERS)}
GRID = np.round(np.arange(.02, .91, .02), 2)
SEQ_COLS = ['time_to_tca', 'log_miss', 'mre', 'log_t', 'log_c', 'l_r', 'l_t', 'l_n', 'o_d', 'o_p', 'o_r']


def reduced():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), PCA(n_components=.99, svd_solver='full', random_state=SEED))


def balanced_weights(y):
    return (len(y) / (3 * np.bincount(y, minlength=3)))[y]


def fit_trees(model, a, b, X, y):
    model.fit(X[a], y[a])
    return model.predict_proba(X[b])


def lightgbm_model():
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=.02, num_leaves=15, min_child_samples=20, subsample=.8, subsample_freq=1,
                              colsample_bytree=.6, reg_lambda=5, class_weight='balanced', random_state=SEED, verbose=-1)


def run_elliptic(a, b, X, y):
    prep = reduced().fit(X[a])
    Pa, Pb = prep.transform(X[a]), prep.transform(X[b])
    safe = y[a] == CLS['SAFE']
    env = EllipticEnvelope(contamination=.02, support_fraction=.9, random_state=SEED).fit(Pa[safe][:30000])
    sa, sb = -env.decision_function(Pa), -env.decision_function(Pb)
    p = np.column_stack([IsotonicRegression(out_of_bounds='clip').fit(sa, y[a] == k).predict(sb) for k in range(3)])
    return p / np.maximum(p.sum(1, keepdims=True), 1e-9)


def run_linear(a, b, X, y):
    prep = reduced().fit(X[a])
    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(prep.transform(X[a]), y[a])
    return m.predict_proba(prep.transform(X[b]))


def run_lgb_pca(a, b, X, y):
    prep = reduced().fit(X[a])
    return lightgbm_model().fit(prep.transform(X[a]), y[a]).predict_proba(prep.transform(X[b]))


def run_rf(a, b, X, y):
    imp = SimpleImputer(strategy='median').fit(X[a])
    m = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=3, max_features=.5, class_weight='balanced_subsample',
                               n_jobs=-1, random_state=SEED).fit(imp.transform(X[a]), y[a])
    return m.predict_proba(imp.transform(X[b]))


def run_xgb(a, b, X, y):
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=.03, max_depth=5, subsample=.8, colsample_bytree=.7, tree_method='hist',
                          n_jobs=-1, random_state=SEED).fit(X[a], y[a], sample_weight=balanced_weights(y[a]))
    return m.predict_proba(X[b])


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(.2), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 3))

    def forward(self, x, n=None):
        return self.net(x)


class GRUNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru, self.head = nn.GRU(d, 48, batch_first=True), nn.Linear(48 + d, 3)

    def forward(self, x, n):
        o, _ = self.gru(x)
        r = torch.arange(len(x))
        return self.head(torch.cat([o[r, n - 1], x[r, n - 1]], 1))


class CausalCNN(nn.Module):
    def __init__(self, d, c=48, k=3):
        super().__init__()
        self.k = k
        self.convs = nn.ModuleList([nn.Conv1d(d, c, k, dilation=1), nn.Conv1d(c, c, k, dilation=2), nn.Conv1d(c, c, k, dilation=4)])
        self.head = nn.Linear(c + d, 3)

    def forward(self, x, n):
        h = x.transpose(1, 2)
        for conv in self.convs:
            h = torch.relu(conv(nn.functional.pad(h, ((self.k - 1) * conv.dilation[0], 0))))
        h = h.transpose(1, 2)
        r = torch.arange(len(x))
        return self.head(torch.cat([h[r, n - 1], x[r, n - 1]], 1))


def train_net(net, x, n, y, epochs, bs):
    torch.manual_seed(SEED)
    weights = torch.tensor(len(y) / (3 * np.bincount(y.numpy(), minlength=3)), dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(x))
        for k in range(0, len(order), bs):
            i = order[k:k + bs]
            loss = nn.functional.cross_entropy(net(x[i], n[i]), y[i], weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def run_mlp(a, b, X, y):
    prep = reduced().fit(X[a])
    xa = torch.tensor(prep.transform(X[a]), dtype=torch.float32)
    xb = torch.tensor(prep.transform(X[b]), dtype=torch.float32)
    net = train_net(MLP(xa.shape[1]), xa, torch.zeros(len(xa), dtype=torch.long), torch.tensor(y[a]), 40, 128)
    with torch.no_grad():
        return net(xb).softmax(1).numpy()


def sequences(raw, ids):
    d = raw[(raw.time_to_tca >= 2) & raw.event_id.isin(ids)].sort_values(['event_id', 'time_to_tca'], ascending=[True, False]).copy()
    d['log_miss'] = np.log10(d.miss_distance.clip(lower=1))
    d['mre'] = d.max_risk_estimate
    d['log_t'] = np.log10(d.t_sigma_r)
    d['log_c'] = np.log10(d.c_sigma_r.fillna(d.c_sigma_r.median()))
    for new, axis in [('l_r', 'r'), ('l_t', 't'), ('l_n', 'n')]:
        d[new] = np.log10(d[f'relative_position_{axis}'].abs() + 1)
    d['o_d'], d['o_p'], d['o_r'] = (d.c_object_type == 'DEBRIS') * 1., (d.c_object_type == 'PAYLOAD') * 1., (d.c_object_type == 'ROCKET BODY') * 1.
    values = d[SEQ_COLS].astype(float).fillna(0).values
    place = {e: i for i, e in enumerate(ids)}
    S = np.zeros((len(ids), d.groupby('event_id').size().max(), len(SEQ_COLS)), np.float32)
    lengths = np.zeros(len(ids), int)
    for e, row in zip(d.event_id.values, values):
        i = place[e]
        S[i, lengths[i]] = row
        lengths[i] += 1
    return S, lengths


def run_seq(cls, S, lengths):
    def run(a, b, X, y):
        filled = np.concatenate([S[i, :lengths[i]] for i in a])
        mu, sd = filled.mean(0), filled.std(0) + 1e-6
        x = torch.tensor((S - mu) / sd, dtype=torch.float32) * torch.tensor(
            (np.arange(S.shape[1])[None, :] < lengths[:, None])[..., None], dtype=torch.float32)
        n = torch.tensor(lengths)
        net = train_net(cls(S.shape[2]), x[a], n[a], torch.tensor(y[a]), 25, 128)
        with torch.no_grad():
            return net(x[b], n[b]).softmax(1).numpy()
    return run


def crossfit(prob_crit, base, y, folds, scope):
    """Critical threshold for each fold is the best F2 on the other folds."""
    out = np.empty(len(y), dtype=object)
    used = []
    for _, b in folds:
        other = np.setdiff1d(np.arange(len(y)), b)
        other = other[scope[other]]
        f2 = [fbeta_score(y[other] == 'CRITICAL', (prob_crit[other] >= t) | (base[other] == 'CRITICAL'), beta=2) for t in GRID]
        t = GRID[int(np.argmax(f2))]
        used.append(t)
        out[b] = np.where(prob_crit[b] >= t, 'CRITICAL', base[b])
    return out, float(np.median(used))


def main():
    labels = pd.read_excel(LABELS, sheet_name='train_event_labels', header=2).set_index('event_id')
    train = pd.read_csv('data/train_data.csv', usecols=RECOMMENDED + ['risk'])
    Xf = event_features(train)
    ev = labels.loc[Xf.index]
    y_txt = ev.final_tier.str.upper().values
    ids = Xf.index.values
    change = np.log10(ev.final_miss_distance_m.clip(lower=1).values) - Xf.log_miss.values
    dev, _ = next(GroupShuffleSplit(1, test_size=.2, random_state=SEED).split(ids, groups=ids))
    ids_d, X_d, y_d = ids[dev], Xf.iloc[dev], y_txt[dev]
    y_num = np.array([CLS[t] for t in y_d])
    has_later = ev.final_time_to_tca_days.values[dev] < 2
    test_like = ev.final_time_to_tca_days.values[dev] < 1
    folds = list(GroupKFold(5).split(X_d, groups=ids_d))
    print('development events', len(dev), 'test-like', int(test_like.sum()), 'critical', int((y_d[test_like] == 'CRITICAL').sum()), flush=True)

    forecast = oof_predict(X_d, y_d, change[dev], ids_d)
    distance = distance_tier(forecast.miss_mid.values)
    S, lengths = sequences(train, ids_d)
    Xn = X_d.values.astype(float)
    runners = {'Anomaly detector': run_elliptic, 'Linear model': run_linear, 'Random forest': run_rf, 'XGBoost': run_xgb,
               'LightGBM with PCA': run_lgb_pca, 'LightGBM': lambda a, b, X, y: fit_trees(lightgbm_model(), a, b, X, y),
               'Neural network (MLP)': run_mlp, 'Sequence network (GRU)': run_seq(GRUNet, S, lengths),
               'Convolutional network (CNN)': run_seq(CausalCNN, S, lengths)}
    probs = {}
    for name, run in runners.items():
        p = np.zeros((len(dev), 3))
        for a, b in folds:
            p[b] = run(a, b, Xn, y_num)
        probs[name] = p
        print('done', name, flush=True)

    def rank_key(p):
        pick = np.array(TIERS)[p.argmax(1)]
        return fbeta_score(y_d[test_like] == 'CRITICAL', pick[test_like] == 'CRITICAL', beta=2)
    top3 = sorted(probs, key=lambda k: rank_key(probs[k]), reverse=True)[:3]
    probs['Ensemble of top three'] = np.mean([probs[k] for k in top3], 0)
    print('ensemble of', top3, flush=True)

    rows = []

    def add(name, mode, pred, threshold=np.nan):
        s = score(y_d[test_like], pred[test_like])
        f2 = fbeta_score(y_d[test_like] == 'CRITICAL', pred[test_like] == 'CRITICAL', beta=2)
        rows.append({'model': name, 'mode': mode, 'f2_critical': f2, **s, 'threshold': threshold})

    last = train[train.time_to_tca >= 2].sort_values('time_to_tca').groupby('event_id').head(1).set_index('event_id').loc[ids_d]
    carried = np.where((last.risk >= -4) | (last.miss_distance < 1000), 'CRITICAL',
                       np.where((last.risk >= -6) | (last.miss_distance <= 5000), 'ELEVATED', 'SAFE'))
    add('Current tier carried forward', 'benchmark', carried)
    add('Distance forecast', 'benchmark', distance)
    tier_names = np.array(TIERS)
    for name, p in probs.items():
        alone, t1 = crossfit(p[:, CLS['CRITICAL']], tier_names[p[:, :2].argmax(1)], y_d, folds, has_later)
        add(name, 'alone', alone, t1)
        both, t2 = crossfit(p[:, CLS['CRITICAL']], distance, y_d, folds, has_later)
        add(name, 'with forecast', both, t2)
    table = pd.DataFrame(rows).sort_values('f2_critical', ascending=False)
    table.to_csv(f'{OUT}/final_tier_model_comparison.csv', index=False)
    print(table.round(3).to_string(index=False), flush=True)

    # is the best really better than the current LightGBM system?
    best = table[table['mode'] != 'benchmark'].iloc[0]
    p = probs[best.model]
    bp = crossfit(p[:, 2], distance if best['mode'] == 'with forecast' else tier_names[p[:, :2].argmax(1)], y_d, folds, has_later)[0]
    cp = crossfit(probs['LightGBM'][:, 2], distance, y_d, folds, has_later)[0]
    rng, diffs = np.random.default_rng(SEED), []
    yy, bb, cc = y_d[test_like], bp[test_like], cp[test_like]
    for _ in range(1000):
        i = rng.integers(0, len(yy), len(yy))
        diffs.append(fbeta_score(yy[i] == 'CRITICAL', bb[i] == 'CRITICAL', beta=2) - fbeta_score(yy[i] == 'CRITICAL', cc[i] == 'CRITICAL', beta=2))
    print(f'best: {best.model} ({best["mode"]}); F2 minus LightGBM with forecast, 95 % interval:', np.percentile(diffs, [2.5, 97.5]).round(3).tolist())


if __name__ == '__main__':
    main()
