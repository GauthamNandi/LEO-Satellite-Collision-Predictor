"""Official Challenge 1 target: predict each event's FINAL tier using only messages >= 2 days before closest approach.

Run after collision_risk_pipeline.ipynb (which writes the message-level estimates the dashboard also shows):
    python final_tier_pipeline.py

Design (all choices made on development events; the 20 % held-out events are scored once):
  - median quantile regression forecasts the final miss distance, which sets the best-guess tier
  - a classifier estimates the chance of ending Critical; above a threshold the event is escalated to Critical
Test events have no final outcomes, so test output is demo predictions only.
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from risk_utils import TIERS, action_level, reaction_budget_days

warnings.filterwarnings('ignore')
SEED = 42
LABELS = 'data/challenge1_event_labels.xlsx'
OUT = 'outputs'
RECOMMENDED = ['event_id', 'time_to_tca', 'miss_distance', 'relative_speed', 'relative_position_r', 'relative_position_t',
               'relative_position_n', 'c_object_type', 'c_sigma_r', 't_sigma_r', 'max_risk_estimate', 'F10', 'AP']
QUANTILES = {'miss_low': .1, 'miss_mid': .5, 'miss_high': .9}
REVIEW_MIN_CHANCE = .10     # below Critical, but this likely to end Critical: an analyst looks


def slope(t, v):
    return 0.0 if len(t) < 2 or np.ptp(t) == 0 else float(np.polyfit(t, v, 1)[0])


def event_features(raw, c_sigma_fill=None):
    """One row per event, built only from messages >= 2 days before closest approach."""
    d = raw[raw.time_to_tca >= 2].sort_values(['event_id', 'time_to_tca'], ascending=[True, False]).copy()
    d['log_miss'] = np.log10(d.miss_distance.clip(lower=1))
    d['log_t_sig'] = np.log10(d.t_sigma_r.clip(lower=1e-3))
    fill = d.c_sigma_r.median() if c_sigma_fill is None else c_sigma_fill
    d['log_c_sig'] = np.log10(d.c_sigma_r.fillna(fill).clip(lower=1e-3))
    rows = {}
    for eid, g in d.groupby('event_id', sort=False):
        l, t = g.iloc[-1], g.time_to_tca.values
        r = dict(n_msgs=len(g), last_t=l.time_to_tca, span=t[0] - t[-1], log_miss=l.log_miss, log_speed=np.log10(max(l.relative_speed, 1)),
                 log_t_sig=l.log_t_sig, log_c_sig=l.log_c_sig, mre=l.max_risk_estimate, F10=l.F10, AP=l.AP,
                 log_r=np.log10(abs(l.relative_position_r) + 1), log_t=np.log10(abs(l.relative_position_t) + 1),
                 log_n=np.log10(abs(l.relative_position_n) + 1), miss_min=g.log_miss.min(), miss_max=g.log_miss.max(),
                 miss_std=g.log_miss.std() if len(g) > 1 else 0., miss_slope=slope(t, g.log_miss.values),
                 mre_max=g.max_risk_estimate.max(), mre_first=g.max_risk_estimate.iloc[0], mre_slope=slope(t, g.max_risk_estimate.values),
                 sig_slope=slope(t, g.log_c_sig.values), sig_drop=g.log_c_sig.iloc[0] - l.log_c_sig,
                 miss_last_jump=l.log_miss - (g.log_miss.iloc[-2] if len(g) > 1 else l.log_miss),
                 miss_over_sig=l.log_miss - np.log10(np.hypot(10 ** l.log_t_sig, 10 ** l.log_c_sig)),
                 to_1km=l.log_miss - 3, to_5km=l.log_miss - np.log10(5000))
        for o in ['DEBRIS', 'PAYLOAD', 'ROCKET BODY', 'UNKNOWN']:
            r['o_' + o.replace(' ', '_')] = int(l.c_object_type == o)
        rows[eid] = r
    return pd.DataFrame.from_dict(rows, orient='index')


def fit(X, cls, miss_change):
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=.02, num_leaves=15, min_child_samples=20, subsample=.8, subsample_freq=1,
                             colsample_bytree=.6, reg_lambda=5, class_weight='balanced', random_state=SEED, verbose=-1).fit(X, cls)
    reg = {k: lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=300, learning_rate=.03, num_leaves=15, min_child_samples=30,
                                subsample=.8, subsample_freq=1, colsample_bytree=.7, random_state=SEED, verbose=-1).fit(X, miss_change)
           for k, q in QUANTILES.items()}
    return clf, reg


def predict(models, X):
    clf, reg = models
    out = pd.DataFrame({'raw_critical': clf.predict_proba(X)[:, list(clf.classes_).index('CRITICAL')]}, index=X.index)
    for k, m in reg.items():
        out[k] = 10 ** (X.log_miss.values + m.predict(X))
    return out


def distance_tier(miss):
    return np.where(miss < 1000, 'CRITICAL', np.where(miss <= 5000, 'ELEVATED', 'SAFE'))


def decide(pred, threshold):
    return np.where(pred.raw_critical >= threshold, 'CRITICAL', distance_tier(pred.miss_mid))


def score(y, p):
    return {'overall': f1_score(y, p, labels=TIERS, average='macro'), 'critical_caught': recall_score(y == 'CRITICAL', p == 'CRITICAL'),
            'critical_right': precision_score(y == 'CRITICAL', p == 'CRITICAL', zero_division=0), 'critical_alerts': int((p == 'CRITICAL').sum())}


def oof_predict(X, cls, change, groups, n=5):
    parts = []
    for a, b in GroupKFold(n).split(X, groups=groups):
        parts.append(predict(fit(X.iloc[a], cls[a], change[a]), X.iloc[b]))
    return pd.concat(parts).loc[X.index]


def main():
    labels = pd.read_excel(LABELS, sheet_name='train_event_labels', header=2).set_index('event_id')
    train = pd.read_csv('data/train_data.csv', usecols=RECOMMENDED + ['risk'])
    test = pd.read_csv('data/test_data.csv', usecols=RECOMMENDED)
    X = event_features(train)
    X_test = event_features(test, train[train.time_to_tca >= 2].c_sigma_r.median())
    ev = labels.loc[X.index]
    y = ev.final_tier.str.upper().values
    change = np.log10(ev.final_miss_distance_m.clip(lower=1).values) - X.log_miss.values
    ids = X.index.values
    has_later = ev.final_time_to_tca_days.values < 2     # otherwise the final message is itself in the input
    test_like = ev.final_time_to_tca_days.values < 1     # same shape as the official test set

    dev, hold = next(GroupShuffleSplit(1, test_size=.2, random_state=SEED).split(ids, groups=ids))
    print(f'events: {len(ids):,} usable, {len(dev):,} development, {len(hold):,} held out, {has_later[hold].sum():,} held out with a later message')

    # choices on development events only
    oof = oof_predict(X.iloc[dev], y[dev], change[dev], ids[dev])
    m = has_later[dev]
    grid = np.round(np.arange(.1, .61, .02), 2)
    f2 = [fbeta_score(y[dev][m] == 'CRITICAL', decide(oof[m], t) == 'CRITICAL', beta=2) for t in grid]
    threshold = float(grid[int(np.argmax(f2))])
    print('critical threshold chosen on development events:', threshold)

    # one score on the held-out events
    h = predict(fit(X.iloc[dev], y[dev], change[dev]), X.iloc[hold])
    last = train[train.time_to_tca >= 2].sort_values('time_to_tca').groupby('event_id').head(1).set_index('event_id').loc[ids]
    carried = np.where((last.risk >= -4) | (last.miss_distance < 1000), 'CRITICAL',
                       np.where((last.risk >= -6) | (last.miss_distance <= 5000), 'ELEVATED', 'SAFE'))
    systems = {'Current tier carried forward': carried[hold], 'Distance forecast only': distance_tier(h.miss_mid.values),
               'Final system': decide(h, threshold), 'Worst-case figure and distance': np.where(
                   (last.max_risk_estimate.values[hold] >= -4) | (last.miss_distance.values[hold] < 1000), 'CRITICAL',
                   np.where((last.max_risk_estimate.values[hold] >= -6) | (last.miss_distance.values[hold] <= 5000), 'ELEVATED', 'SAFE'))}
    scopes = {'Events with a later message': has_later[hold], 'Test-like events': test_like[hold], 'All usable events': np.ones(len(hold), bool)}
    rows = [{'scope': s, 'system': n, **score(y[hold][k], p[k])} for s, k in scopes.items() for n, p in systems.items()]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(f'{OUT}/final_tier_metrics.csv', index=False)
    print(metrics.round(3).to_string(index=False))

    # paired bootstrap, final system minus benchmark, on test-like held-out events
    k = test_like[hold]
    yk, a, b = y[hold][k], systems['Final system'][k], systems['Current tier carried forward'][k]
    rng, diffs = np.random.default_rng(SEED), []
    for _ in range(1000):
        i = rng.integers(0, len(yk), len(yk))
        sa, sb = score(yk[i], a[i]), score(yk[i], b[i])
        diffs.append([sa['overall'] - sb['overall'], sa['critical_caught'] - sb['critical_caught'], sa['critical_right'] - sb['critical_right']])
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0).round(3).tolist()
    print('95 % interval, final system minus carried forward: overall', (lo[0], hi[0]), 'caught', (lo[1], hi[1]), 'right', (lo[2], hi[2]))
    pd.DataFrame(confusion_matrix(yk, a, labels=TIERS), TIERS, TIERS).to_csv(f'{OUT}/final_tier_confusion.csv')

    # refit on every usable event, then make demo predictions for the test events
    all_oof = oof_predict(X, y, change, ids)
    calibrate = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(all_oof.raw_critical, y == 'CRITICAL')
    pred = predict(fit(X, y, change), X_test)
    pred['chance_critical'] = calibrate.predict(pred.raw_critical)
    pred['forecast_tier'] = decide(pred, threshold)
    pred['last_t'] = X_test.last_t
    review = (pred.forecast_tier != 'CRITICAL') & (pred.chance_critical >= REVIEW_MIN_CHANCE)
    pred['action'] = action_level(pred.forecast_tier, pred.last_t, review, pred.miss_mid)
    pred.loc[review, 'action'] = 'REVIEW'       # action_level only reviews Elevated; a likely-Critical Safe forecast needs it too
    pred['reaction_budget_days'] = reaction_budget_days(pred.last_t)
    pred.drop(columns='raw_critical').rename_axis('event_id').to_csv(f'{OUT}/final_tier_test.csv')
    print('test forecast:', pred.forecast_tier.value_counts().to_dict(), pred.action.value_counts().to_dict())
    with open(f'{OUT}/final_tier_config.json', 'w') as f:
        json.dump({'critical_threshold': threshold, 'review_min_chance': REVIEW_MIN_CHANCE, 'features': list(X.columns),
                   'interval_vs_carried': {'overall': [lo[0], hi[0]], 'critical_caught': [lo[1], hi[1]], 'critical_right': [lo[2], hi[2]]}}, f, indent=2, default=float)


if __name__ == '__main__':
    main()
