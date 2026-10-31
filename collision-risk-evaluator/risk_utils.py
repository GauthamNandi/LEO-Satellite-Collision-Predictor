"""Shared logic for the collision-risk notebook and dashboard.

Tier rules, the hybrid safety net, the human/autonomous urgency framework,
manoeuvre delta-v estimates and a Monte Carlo collision-probability what-if.
"""
import numpy as np

TIERS = ['SAFE', 'ELEVATED', 'CRITICAL']
TIER_COLORS = {'SAFE': '#2a9d8f', 'ELEVATED': '#c28226', 'CRITICAL': '#e0527f'}

# ---------------------------------------------------------------- tiers
def tier_from(risk, miss_m, crit=-4.0, elev=-6.0):
    """Challenge-1 tier rule. risk = log10(Pc), miss distance in metres.
    crit/elev are the log10(Pc) cut-offs (brief: -4 and -6; may be tuned for *predicted* risk)."""
    risk, miss_m = np.asarray(risk, float), np.asarray(miss_m, float)
    return np.where((risk >= crit) | (miss_m < 1000), 'CRITICAL',
           np.where((risk >= elev) | (miss_m <= 5000), 'ELEVATED', 'SAFE'))


def pc_tier(risk, crit=-4.0, elev=-6.0):
    """Tier from Pc alone (ignores miss distance)."""
    risk = np.asarray(risk, float)
    return np.where(risk >= crit, 'CRITICAL', np.where(risk >= elev, 'ELEVATED', 'SAFE'))


def hybrid_tier(pred_risk, miss_m, max_risk_estimate, crit=-4.0, elev=-6.0, net_floor=None):
    """ML tier, escalated to CRITICAL when the CDM's own upper-bound Pc >= 1e-4.
    net_floor: if set, only escalate when the ML prediction is also >= net_floor (selective safety net)."""
    t = tier_from(pred_risk, miss_m, crit, elev).astype(object)
    esc = np.asarray(max_risk_estimate, float) >= -4
    if net_floor is not None:
        esc &= np.asarray(pred_risk, float) >= net_floor
    t[esc] = 'CRITICAL'
    return t.astype(str)

# ---------------------------------------------------------------- human vs autonomous
# Operational assumptions (configurable). Typical CA practice: a manoeuvre needs planning,
# screening of the post-manoeuvre orbit, command uplink at a ground-station pass, then the burn.
EXEC_LEAD_DAYS = 0.5    # time needed after a "go" decision to uplink + execute the burn
HUMAN_MIN_DAYS = 1.0    # time an analyst team needs for a considered decision

ACTIONS = ['AUTONOMOUS', 'URGENT', 'PLAN', 'REVIEW', 'MONITOR', 'NO ACTION']
ACTION_COLORS = {'AUTONOMOUS': '#7b2cbf', 'URGENT': '#d1495b', 'PLAN': '#f08c00',
                 'REVIEW': '#e9a23b', 'MONITOR': '#74c0fc', 'NO ACTION': '#2a9d8f'}
ACTION_TEXT = {
    'AUTONOMOUS': 'No time for a human decision: execute the pre-planned avoidance manoeuvre automatically; operator keeps a veto.',
    'URGENT': 'Human decision needed now: an avoidance plan is prepared automatically and armed.',
    'PLAN': 'There is time for a considered decision, so an analyst plans a manoeuvre.',
    'REVIEW': 'The model is unsure, so an analyst should check this event.',
    'MONITOR': 'No action yet. Check again when the next update arrives.',
    'NO ACTION': 'This event is safe.',
}
ACTION_LABELS = {'AUTONOMOUS': 'Automatic manoeuvre', 'URGENT': 'Decide now', 'PLAN': 'Plan manoeuvre',
                 'REVIEW': 'Analyst review', 'MONITOR': 'Monitor', 'NO ACTION': 'No action'}
OBJECT_LABELS = {'DEBRIS': 'Debris', 'PAYLOAD': 'Satellite', 'ROCKET BODY': 'Rocket body', 'UNKNOWN': 'Unknown', 'TBA': 'Unknown'}
TARGET_PC = 3.1e-6      # NASA handbook: cut Pc by at least 1.5 orders of magnitude (1e-4 -> 3.1e-6)


def reaction_budget_days(time_to_tca):
    """Time a human has left to decide before the manoeuvre must be committed."""
    return np.asarray(time_to_tca, float) - EXEC_LEAD_DAYS


def action_level(tier, time_to_tca, review_flag, miss_m):
    tier, review_flag = np.asarray(tier), np.asarray(review_flag, bool)
    budget = reaction_budget_days(time_to_tca)
    crit = tier == 'CRITICAL'
    confident = (~review_flag) | (np.asarray(miss_m, float) < 1000)
    return np.select(
        [crit & (budget <= 0) & confident,          # too late for a human, model sure
         crit & (budget < HUMAN_MIN_DAYS),          # short window, or too late but uncertain
         crit,
         (tier == 'ELEVATED') & review_flag,
         tier == 'ELEVATED'],
        ['AUTONOMOUS', 'URGENT', 'PLAN', 'REVIEW', 'MONITOR'], 'NO ACTION')

# ---------------------------------------------------------------- manoeuvre physics
def dv_required_mm_s(separation_m, lead_days):
    """Along-track delta-v needed to build up `separation_m` of along-track displacement by TCA.
    Clohessy-Wiltshire secular drift: delta_s ~= 3 * dv * t  (valid when t >> orbital period, ~1.5 h in LEO)."""
    t = np.maximum(np.asarray(lead_days, float), 1e-3) * 86400.0
    return np.asarray(separation_m, float) / (3.0 * t) * 1e3


def along_track_shift_m(dv_mm_s, lead_days):
    return 3.0 * (np.asarray(dv_mm_s, float) / 1e3) * np.asarray(lead_days, float) * 86400.0

# ---------------------------------------------------------------- plain-language event advice
def chance_text(pc):
    """Collision probability as '1 in 12,000' (two significant figures)."""
    if not pc > 0 or pc < 1e-9:
        return 'Below 1 in a billion'
    if pc >= 0.5:
        return 'Above 1 in 2'
    return f'1 in {float(f"{1 / pc:.2g}"):,.0f}'


def duration_text(days):
    """Days as '5 days 6 hours' or '18 hours'."""
    hours = int(round(max(float(days), 0) * 24))
    d, h = divmod(hours, 24)
    unit = lambda n, name: f'{n} {name}' + ('' if n == 1 else 's')
    if d == 0:
        return unit(h, 'hour')
    return unit(d, 'day') + (f' {unit(h, "hour")}' if h else '')


def chance_percent(p):
    return 'Under 1%' if p < .01 else f'{p:.0%}'


def recommended_actions(last, history):
    """Reasons and ordered next steps for one event, as (why, steps).
    `last` is the latest message of the event merged with its final-tier forecast, `history` all its messages, oldest first.
    The critical-tier steps follow the NASA conjunction assessment handbook: reduce Pc by at least
    1.5 orders of magnitude, screen the new trajectory, coordinate with active satellites, keep an abort."""
    trend = last.pred_log10_pc - history.pred_log10_pc.iloc[0]
    why = [f'Final distance is forecast at about {last.miss_mid / 1e3:,.1f} km (likely {last.miss_low / 1e3:,.1f} to {last.miss_high / 1e3:,.1f} km).',
           f'Chance of ending Critical: {chance_percent(last.chance_critical)}.']
    if last.max_risk_estimate >= -4:
        why.append(f'The worst-case collision chance is {chance_text(10.0 ** last.max_risk_estimate)}.')
    if len(history) > 1 and abs(trend) >= 0.5:
        why.append('Risk has been rising across updates.' if trend > 0 else 'Risk has been falling across updates.')

    escalate = 'Move to manoeuvre planning if the chance of ending Critical rises or the distance drops below 1 km.'
    if last.action == 'NO ACTION':
        return why, ['Keep routine tracking. No manoeuvre is needed.']
    if last.action == 'MONITOR':
        return why, ['Check again when the next update arrives.', escalate]
    if last.action == 'REVIEW':
        return why, ['Ask an analyst to check the quality of the tracking data behind this update.',
                     'Compare the forecast with the next update.', escalate]

    speed = float(dv_required_mm_s(max(5000 - last.miss_mid, 500), last.time_to_tca))
    who = {'PAYLOAD': 'The other object may be an active satellite. Contact its operator and agree who moves.',
           'DEBRIS': 'The other object is debris and cannot move, so the manoeuvre is ours to make.',
           'ROCKET BODY': 'The other object is a rocket body and cannot move, so the manoeuvre is ours to make.'}.get(
        last.c_object_type, 'The other object is unidentified. Ask the tracking provider what it is, and assume it will not move.')
    plan = (f'Prepare a speed change of about {speed:.1f} mm/s along the direction of travel. '
            f'Aim for a collision chance below {chance_text(TARGET_PC)}.')
    screen = 'Check the new path for other close approaches, then share it with the tracking provider.'
    if last.action == 'AUTONOMOUS':
        return why, ['Let the prepared manoeuvre run unless an operator stops it.', 'Keep the abort command ready.', who, screen]
    first = ('Treat this as a real risk and start manoeuvre planning now.' if last.chance_critical >= .5
             else 'Confirm with the next update before using fuel, because this is a cautious alert.')
    if last.action == 'URGENT':
        first = 'A person must decide now. Do not wait for another update.'
    return why, [first, who, plan, screen]

# ---------------------------------------------------------------- Monte Carlo Pc
# Only radial sigmas are allowed inputs, so the along-track / normal sigmas and the hard-body radius are
# calibrated against the dataset's Pc on training data (notebook section 10.6). Defaults = calibrated values.
HBR_M = 50.0            # combined hard-body radius (m)
SIGMA_T_FACTOR = 100.0  # along-track sigma = factor x combined radial sigma
SIGMA_N_FACTOR = 50.0   # normal sigma = factor x combined radial sigma
SIGMA_CAP_M = 1e5


def mc_pc(rel_rtn_m, t_sigma_r, c_sigma_r, shift_t_m=0.0, n=1_000_000, hbr=HBR_M,
          t_factor=SIGMA_T_FACTOR, n_factor=SIGMA_N_FACTOR, seed=0, chunk=500_000):
    """Monte Carlo collision probability at TCA.
    Samples the relative position (RTN, metres) from a Gaussian centred on the CDM's miss vector,
    optionally shifted along-track by a manoeuvre, and counts samples inside the hard-body sphere.
    Returns (pc, upper_bound_flag). If no sample hits, returns the rule-of-three bound 3/n.
    Approximation: treats the sampled vector as the closest-approach vector (no encounter-plane projection,
    since relative-velocity components are not allowed inputs)."""
    rng = np.random.default_rng(seed)
    sr = min(float(np.hypot(t_sigma_r, np.nan_to_num(c_sigma_r, nan=t_sigma_r))), SIGMA_CAP_M)
    sig = np.array([sr, t_factor * sr, n_factor * sr])
    mu = np.asarray(rel_rtn_m, float).copy()
    mu[1] += np.sign(mu[1] if mu[1] != 0 else 1.0) * shift_t_m      # push away along-track
    hits, done = 0, 0
    while done < n:
        k = min(chunk, n - done)
        x = rng.standard_normal((k, 3)) * sig + mu
        hits += int((np.einsum('ij,ij->i', x, x) < hbr ** 2).sum())
        done += k
    if hits == 0:
        return 3.0 / n, True
    return hits / n, False


def pc_analytic_log10(rel_rtn_m, t_sigma_r, c_sigma_r, hbr=HBR_M, t_factor=SIGMA_T_FACTOR, n_factor=SIGMA_N_FACTOR):
    """Small-object approximation of the same model as mc_pc (vectorised):
    Pc ~= sphere volume x Gaussian density at the origin. Used to calibrate the assumed sigma factors."""
    rel = np.atleast_2d(np.asarray(rel_rtn_m, float))
    sr = np.minimum(np.hypot(t_sigma_r, np.nan_to_num(c_sigma_r, nan=np.nanmedian(c_sigma_r))), SIGMA_CAP_M)
    sig = np.stack([sr, t_factor * sr, n_factor * sr], 1)
    q = ((rel / sig) ** 2).sum(1)
    log_pc = np.log(4 / 3 * np.pi * hbr ** 3) - 1.5 * np.log(2 * np.pi) - np.log(sig).sum(1) - 0.5 * q
    return log_pc / np.log(10)
