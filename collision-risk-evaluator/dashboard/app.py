"""Collision Risk Evaluator — operations dashboard.

Run from the repo root after executing the notebook:
    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from risk_utils import (ACTION_COLORS, ACTION_TEXT, ACTIONS, EXEC_LEAD_DAYS, HUMAN_MIN_DAYS, TIER_COLORS, TIERS,  # noqa: E402
                        along_track_shift_m, dv_required_mm_s, mc_pc)

OUT = BASE / 'outputs'
st.set_page_config(page_title='Collision Risk Evaluator', page_icon='🛰️', layout='wide')
st.markdown("""<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.action-card {border-radius: 10px; padding: 14px 18px; color: white; margin-bottom: 10px;}
.action-card h3 {margin: 0 0 4px 0; color: white;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    need = ['predictions_test.csv', 'event_summary_test.csv', 'config.json']
    missing = [f for f in need if not (OUT / f).exists()]
    if missing:
        return None
    p = pd.read_csv(OUT / 'predictions_test.csv')
    ev = pd.read_csv(OUT / 'event_summary_test.csv')
    cfg = json.load(open(OUT / 'config.json'))
    opt = lambda f, **kw: pd.read_csv(OUT / f, **kw) if (OUT / f).exists() else None
    return dict(p=p, ev=ev, cfg=cfg, comp=opt('final_comparison_test.csv', index_col=0), metrics=opt('test_metrics.csv', index_col=0),
                w24=opt('warnings_24h_train.csv'), trans=opt('tier_transition_train.csv'), esc=opt('escalation_timing_train.csv'))


D = load()
if D is None:
    st.error('No results found in `outputs/`. Run `collision_risk_pipeline.ipynb` first.')
    st.stop()
p, ev, cfg = D['p'], D['ev'], D['cfg']
SEV = {a: i for i, a in enumerate(ACTIONS)}

# ------------------------------------------------------------------ header
st.title('🛰️ Satellite Collision Risk Evaluator')
st.caption(f"Model: **{cfg['best_model']}** + upper-bound safety net · tier cut-offs on predicted log10(Pc): "
           f"Critical ≥ {cfg['crit_cut']}, Elevated ≥ {cfg['elev_cut']} · scope: test set, {p.event_id.nunique():,} conjunction events, "
           f"{len(p):,} CDMs (all ≥ 2 days before TCA)")

tab_ops, tab_ev, tab_ins, tab_model = st.tabs(['Operations overview', 'Event explorer', '24 h warnings & insights', 'Model performance'])

latest = p.sort_values('time_to_tca').groupby('event_id').head(1).copy()       # most recent CDM per event
latest['sev'] = latest.action.map(SEV)
first_flag = ev.set_index('event_id').first_critical_flag_days

# ================================================================== OPERATIONS
with tab_ops:
    c = st.columns(6)
    c[0].metric('Events tracked', f'{len(latest):,}')
    c[1].metric('Critical now', int((latest.final_tier == 'CRITICAL').sum()))
    c[2].metric('Autonomous / urgent', f"{int((latest.action == 'AUTONOMOUS').sum())} / {int((latest.action == 'URGENT').sum())}")
    c[3].metric('Manoeuvre planning', int((latest.action == 'PLAN').sum()))
    c[4].metric('Analyst review queue', int((latest.action == 'REVIEW').sum()))
    c[5].metric('Median warning lead time', f'{first_flag.median():.1f} d')

    left, right = st.columns([1.55, 1])
    with left:
        st.subheader('Priority queue')
        f1, f2 = st.columns(2)
        acts = f1.multiselect('Action', ACTIONS, default=['AUTONOMOUS', 'URGENT', 'PLAN', 'REVIEW'])
        objs = f2.multiselect('Object type', sorted(latest.c_object_type.unique()), default=sorted(latest.c_object_type.unique()))
        q = latest[latest.action.isin(acts) & latest.c_object_type.isin(objs)].sort_values(['sev', 'pred_log10_pc'], ascending=[True, False])
        show = q[['event_id', 'action', 'final_tier', 'c_object_type', 'pred_pc', 'miss_distance', 'time_to_tca', 'reaction_budget_days']].rename(
            columns={'event_id': 'event', 'final_tier': 'tier', 'c_object_type': 'object', 'pred_pc': 'predicted Pc',
                     'miss_distance': 'miss (m)', 'time_to_tca': 'TCA in (d)', 'reaction_budget_days': 'decide within (d)'})
        st.dataframe(show.style.format({'predicted Pc': '{:.1e}', 'miss (m)': '{:,.0f}', 'TCA in (d)': '{:.2f}', 'decide within (d)': '{:.2f}'})
                     .apply(lambda s: [f'background-color: {ACTION_COLORS[v]}33' for v in s], subset=['action']),
                     height=420, width='stretch', hide_index=True)
        st.caption(f'{len(q)} events · "decide within" = time to TCA − {EXEC_LEAD_DAYS} d needed to uplink and execute a burn')
    with right:
        st.subheader('Where the events stand')
        ac = latest.action.value_counts().reindex(ACTIONS).fillna(0).reset_index()
        ac.columns = ['action', 'events']
        fig = px.bar(ac, x='events', y='action', orientation='h', color='action', color_discrete_map=ACTION_COLORS, text='events')
        fig.update_layout(showlegend=False, height=260, margin=dict(l=0, r=10, t=10, b=0), yaxis=dict(categoryorder='array', categoryarray=ACTIONS[::-1]))
        st.plotly_chart(fig, width='stretch')
        to = pd.crosstab(latest.c_object_type, latest.final_tier).reindex(columns=TIERS).fillna(0).reset_index().melt(id_vars='c_object_type')
        fig = px.bar(to, y='c_object_type', x='value', color='final_tier', orientation='h', color_discrete_map=TIER_COLORS,
                     labels={'value': 'events', 'c_object_type': ''})
        fig.update_layout(height=250, margin=dict(l=0, r=10, t=10, b=0), legend=dict(orientation='h', y=-0.25, title=''))
        st.plotly_chart(fig, width='stretch')

    st.subheader('Risk map — latest CDM per event')
    fig = px.scatter(latest.assign(miss_km=latest.miss_distance / 1e3), x='miss_km', y='pred_log10_pc', color='final_tier',
                     color_discrete_map=TIER_COLORS, hover_data=['event_id', 'c_object_type', 'action', 'time_to_tca'], log_x=True,
                     labels={'miss_km': 'miss distance (km, log)', 'pred_log10_pc': 'predicted log10(Pc)'}, opacity=.7)
    for y, col in [(-4, TIER_COLORS['CRITICAL']), (-6, TIER_COLORS['ELEVATED'])]:
        fig.add_hline(y=y, line_dash='dash', line_color=col)
    for x, col in [(1, TIER_COLORS['CRITICAL']), (5, TIER_COLORS['ELEVATED'])]:
        fig.add_vline(x=x, line_dash='dash', line_color=col)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')

# ================================================================== EVENT EXPLORER
with tab_ev:
    order = latest.sort_values(['sev', 'pred_log10_pc'], ascending=[True, False])
    labels = {r.event_id: f'Event {r.event_id} · {r.action} · {r.c_object_type} · Pc {r.pred_pc:.1e}' for r in order.itertuples()}
    eid = st.selectbox('Event (sorted by priority)', list(labels), format_func=labels.get)
    g = p[p.event_id == eid].sort_values('time_to_tca', ascending=False)
    last = g.iloc[-1]

    a, b = st.columns([1, 2.2])
    with a:
        col = ACTION_COLORS[last.action]
        st.markdown(f"""<div class="action-card" style="background:{col}"><h3>{last.action}</h3>{ACTION_TEXT[last.action]}</div>""",
                    unsafe_allow_html=True)
        m = st.columns(2)
        m[0].metric('Final tier', last.final_tier)
        m[1].metric('Predicted Pc', f'{last.pred_pc:.1e}')
        m[0].metric('Miss distance', f'{last.miss_distance:,.0f} m')
        m[1].metric('Time to TCA', f'{last.time_to_tca:.2f} d')
        m[0].metric('Decide within', f'{max(last.reaction_budget_days, 0):.2f} d')
        ff = first_flag.get(eid, np.nan)
        m[1].metric('First Critical flag', '—' if pd.isna(ff) else f'{ff:.2f} d before TCA')
        need = max(5000 - last.miss_distance, 500)
        st.caption(f'Δv to open separation to 5 km: **{dv_required_mm_s(need, last.time_to_tca):.2f} mm/s** if executed now, '
                   f'**{dv_required_mm_s(need, 0.5):.1f} mm/s** if left until T−12 h.')
        show_truth = st.toggle('Show ground-truth Pc (evaluation only)', value=False)
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.time_to_tca, y=g.pred_log10_pc, mode='lines+markers', name='predicted log10(Pc)', line=dict(color='#4c6ef5')))
        fig.add_trace(go.Scatter(x=g.time_to_tca, y=g.max_risk_estimate, mode='lines+markers', name='max_risk_estimate (upper bound)',
                                 line=dict(color='#868e96', dash='dot')))
        if show_truth:
            fig.add_trace(go.Scatter(x=g.time_to_tca, y=g.true_log10_pc.clip(lower=-10), mode='lines+markers', name='true log10(Pc)', line=dict(color='black')))
        fig.add_hline(y=-4, line_dash='dash', line_color=TIER_COLORS['CRITICAL'], annotation_text='Critical 1e-4')
        fig.add_hline(y=-6, line_dash='dash', line_color=TIER_COLORS['ELEVATED'], annotation_text='Elevated 1e-6')
        fig.update_layout(title='Collision probability as CDMs arrive', xaxis_title='days to TCA', yaxis_title='log10(Pc)',
                          xaxis_autorange='reversed', height=300, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation='h', y=-0.3))
        st.plotly_chart(fig, width='stretch')
        fig = go.Figure(go.Scatter(x=g.time_to_tca, y=g.miss_distance / 1e3, mode='lines+markers', line=dict(color='#2b2d42'),
                                   marker=dict(color=[TIER_COLORS[t] for t in g.final_tier], size=9), name='miss distance'))
        fig.add_hline(y=1, line_dash='dash', line_color=TIER_COLORS['CRITICAL']); fig.add_hline(y=5, line_dash='dash', line_color=TIER_COLORS['ELEVATED'])
        fig.update_layout(title='Predicted miss distance per CDM (marker colour = tier)', xaxis_title='days to TCA', yaxis_title='km',
                          xaxis_autorange='reversed', height=260, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Encounter geometry (RTN frame)')
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
        fig = go.Figure()
        reach = max(1.2, 1.15 * np.abs(g[['relative_position_r', 'relative_position_t', 'relative_position_n']].values).max() / 1e3)
        for rad, colr, op in [(1, TIER_COLORS['CRITICAL'], .10), (5, TIER_COLORS['ELEVATED'], .04)]:
            if rad > 2.5 * reach:
                continue
            fig.add_trace(go.Surface(x=rad * np.cos(u) * np.sin(v), y=rad * np.sin(u) * np.sin(v), z=rad * np.cos(v), opacity=op,
                                     showscale=False, colorscale=[[0, colr], [1, colr]], name=f'{rad} km', hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=g.relative_position_t / 1e3, y=g.relative_position_n / 1e3, z=g.relative_position_r / 1e3,
                                   mode='lines+markers', marker=dict(size=6, color=g.time_to_tca, colorscale='Viridis_r', colorbar=dict(title='days<br>to TCA')),
                                   line=dict(color='#495057', width=4), name='miss-vector estimate',
                                   text=[f'{t:.2f} d before TCA' for t in g.time_to_tca]))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=4, color='black', symbol='diamond'), name='our satellite'))
        rng = [-reach, reach]
        fig.update_layout(scene=dict(xaxis=dict(title='T (km)', range=rng), yaxis=dict(title='N (km)', range=rng),
                                     zaxis=dict(title='R (km)', range=rng), aspectmode='cube'),
                          height=430, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption('Each point is where one CDM predicts the debris will be, relative to our satellite, at closest approach. '
                   'The track shows how the estimate converges as CDMs arrive (spheres: 1 km and 5 km).')
    with c2:
        st.subheader('Manoeuvre what-if (Monte Carlo)')
        dv = st.slider('Along-track Δv (mm/s)', 0.0, 30.0, 2.0, 0.25)
        lead = st.slider('Burn executed this many days before TCA', 0.1, float(max(0.5, round(last.time_to_tca, 1))),
                         float(min(2.0, round(last.time_to_tca, 1))), 0.1)
        rel = [last.relative_position_r, last.relative_position_t, last.relative_position_n]

        @st.cache_data
        def _mc(rel, ts, cs, shift):
            return mc_pc(list(rel), ts, cs, shift_t_m=shift, n=400_000, seed=1, **cfg.get('mc', {}))

        pc0, b0 = _mc(tuple(rel), last.t_sigma_r, last.c_sigma_r, 0.0)
        shift = float(along_track_shift_m(dv, lead))
        pc1, b1 = _mc(tuple(rel), last.t_sigma_r, last.c_sigma_r, shift)
        k = st.columns(3)
        k[0].metric('Pc, no manoeuvre (MC)', ('< ' if b0 else '') + f'{pc0:.1e}')
        k[1].metric('Pc after burn (MC)', ('< ' if b1 else '') + f'{pc1:.1e}', delta=f'{np.log10(pc1) - np.log10(pc0):+.1f} decades', delta_color='inverse')
        k[2].metric('Added separation', f'{shift / 1e3:.2f} km')
        dvs = np.array([0, 0.5, 1, 2, 4, 8, 15, 30])
        rows = []
        for ld in sorted({3.0, 1.0, 0.5, round(lead, 1)}):
            rows += [{'Δv (mm/s)': d, 'lead': f'{ld} d', 'log10 Pc': np.log10(_mc(tuple(rel), last.t_sigma_r, last.c_sigma_r,
                                                                                float(along_track_shift_m(d, ld)))[0])} for d in dvs]
        fig = px.line(pd.DataFrame(rows), x='Δv (mm/s)', y='log10 Pc', color='lead', markers=True)
        fig.add_hline(y=-4, line_dash='dash', line_color=TIER_COLORS['CRITICAL']); fig.add_hline(y=-6, line_dash='dash', line_color=TIER_COLORS['ELEVATED'])
        fig.add_hline(y=np.log10(3 / 400_000), line_dash='dot', line_color='grey', annotation_text='MC resolution floor', annotation_position='bottom right')
        fig.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation='h', y=-0.3))
        st.plotly_chart(fig, width='stretch')
        mcp = cfg.get('mc', {})
        st.caption(f"Gaussian position uncertainty: σR from the CDM radial sigmas, σT = {mcp.get('t_factor', '?'):g}·σR, σN = {mcp.get('n_factor', '?'):g}·σR, "
                   f"hard-body radius {mcp.get('hbr', '?'):g} m (calibrated on training data); along-track drift ≈ 3·Δv·t. "
                   'Illustrative what-if only: the risk tier comes from the ML model, not this simulation.')

# ================================================================== INSIGHTS
with tab_ins:
    w24, trans, esc = D['w24'], D['trans'], D['esc']
    if w24 is not None:
        st.subheader('Last 24 hours before closest approach (full training history)')
        c = st.columns(4)
        c[0].metric('Events with a CDM in final 24 h', f'{len(w24):,}')
        c[1].metric('Within 1 km (Critical)', int((w24.miss_distance < 1000).sum()))
        c[2].metric('Within 5 km', int((w24.miss_distance <= 5000).sum()))
        c[3].metric('Pc ≥ 1e-4', int((w24.risk >= -4).sum()))
        l, r = st.columns(2)
        with l:
            fig = px.histogram(w24.assign(miss_km=w24.miss_distance / 1e3).query('miss_km < 30'), x='miss_km', nbins=60, color='tier',
                               color_discrete_map=TIER_COLORS, labels={'miss_km': 'miss distance of final CDM (km)'})
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title='Miss distance in the final 24 h', bargap=.05)
            st.plotly_chart(fig, width='stretch')
        with r:
            if trans is not None:
                ct = pd.crosstab(trans.tier_2d, trans.tier_final).reindex(index=TIERS, columns=TIERS).fillna(0)
                fig = px.imshow(ct, text_auto=True, color_continuous_scale='Blues', labels=dict(x='tier in final 24 h', y='tier ≥ 2 days out', color='events'))
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title='How tiers change from 2 days out to the final day')
                st.plotly_chart(fig, width='stretch')
    if esc is not None:
        st.subheader('Does a human have time to react?')
        l, r = st.columns(2)
        with l:
            fig = px.histogram(esc, x='first_crit_tca', nbins=28, color_discrete_sequence=[TIER_COLORS['CRITICAL']],
                               labels={'first_crit_tca': 'days before TCA the event first became Critical'})
            fig.add_vrect(x0=0, x1=EXEC_LEAD_DAYS, fillcolor=ACTION_COLORS['AUTONOMOUS'], opacity=.15, annotation_text='autonomous', line_width=0)
            fig.add_vrect(x0=EXEC_LEAD_DAYS, x1=EXEC_LEAD_DAYS + HUMAN_MIN_DAYS, fillcolor=ACTION_COLORS['URGENT'], opacity=.12, annotation_text='urgent', line_width=0)
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), xaxis_autorange='reversed', title='When events first turn Critical')
            st.plotly_chart(fig, width='stretch')
        with r:
            lead = np.linspace(0.1, 7, 150)
            df = pd.concat([pd.DataFrame({'lead (days)': lead, 'Δv (mm/s)': dv_required_mm_s(s, lead), 'separation': f'+{s // 1000} km'})
                            for s in (1000, 2000, 5000)])
            fig = px.line(df, x='lead (days)', y='Δv (mm/s)', color='separation', log_y=True)
            fig.add_vrect(x0=0, x1=EXEC_LEAD_DAYS, fillcolor=ACTION_COLORS['AUTONOMOUS'], opacity=.15, line_width=0)
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), xaxis_autorange='reversed', title='The cost of waiting: Δv vs lead time')
            st.plotly_chart(fig, width='stretch')
        n = len(esc)
        auto = (esc.first_crit_tca < EXEC_LEAD_DAYS).sum()
        urgent = ((esc.first_crit_tca >= EXEC_LEAD_DAYS) & (esc.first_crit_tca < EXEC_LEAD_DAYS + HUMAN_MIN_DAYS)).sum()
        evx = ev.dropna(subset=['dv_at_first_flag'])
        st.markdown(f"""
**Key insights**
- **{(n - auto - urgent) / n:.0%}** of events that turn Critical do so with ≥ {EXEC_LEAD_DAYS + HUMAN_MIN_DAYS:.1f} days to go — a human has time; median decision budget **{esc.reaction_budget_days.median():.1f} days**.
- **{urgent / n:.1%}** need an urgent human call, and **{auto / n:.1%}** ({auto} events) become Critical inside {EXEC_LEAD_DAYS} days — too late for a human, the case for autonomy with operator veto.
- On test events the system raises the first Critical flag a median **{first_flag.median():.1f} days** before TCA (minimum {first_flag.min():.1f} d).
- Acting at the first flag needs a median **{evx.dv_at_first_flag.median():.1f} mm/s** to reach 5 km separation; waiting until T−12 h needs **{evx.dv_at_12h.median():.0f} mm/s** (≈{(evx.dv_at_12h / evx.dv_at_first_flag).median():.0f}× more).
""")
        st.caption(f'Assumed execution lead {EXEC_LEAD_DAYS} d and human decision time {HUMAN_MIN_DAYS} d (edit in risk_utils.py).')

# ================================================================== MODEL
with tab_model:
    comp, met = D['comp'], D['metrics']
    if comp is not None:
        st.subheader('Final system vs baseline (test set)')
        st.dataframe(comp.style.format({c: '{:.3f}' for c in comp.columns if c != 'critical_alerts'}), width='stretch')
    if met is not None:
        st.subheader('All models (test set, ML tier only)')
        cols = ['macro_f1', 'critical_recall', 'critical_precision', 'pc_macro_f1', 'pc_critical_recall', 'rmse_highrisk']
        st.dataframe(met[cols].sort_values('macro_f1', ascending=False).style.format('{:.3f}').background_gradient(cmap='Greens', subset=['macro_f1', 'pc_macro_f1']),
                     width='stretch')
    for img, cap in [('final_confusion_test.png', 'Final system — confusion matrix'), ('confusion_matrices_test.png', 'Every model — confusion matrices')]:
        if (OUT / img).exists():
            st.image(str(OUT / img), caption=cap)
