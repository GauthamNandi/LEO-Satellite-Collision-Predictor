"""Collision Risk Evaluator — operations dashboard.

Run from the repo root after executing the notebook:
    streamlit run dashboard/app.py

All visible text avoids acronyms: say "closest approach", "collision chance", "update" and "speed change".
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
from risk_utils import (ACTION_COLORS, ACTION_LABELS, ACTION_TEXT, ACTIONS, EXEC_LEAD_DAYS, HUMAN_MIN_DAYS, OBJECT_LABELS,  # noqa: E402
                        TIER_COLORS, TIERS, along_track_shift_m, chance_percent, chance_text, dv_required_mm_s, duration_text, mc_pc,
                        recommended_actions)

OUT = BASE / 'outputs'
st.set_page_config(page_title='Collision Risk Monitor', page_icon='🛰️', layout='wide')
st.markdown("""<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.9rem;}
</style>""", unsafe_allow_html=True)

BADGES = {'AUTONOMOUS': 'violet', 'URGENT': 'red', 'PLAN': 'orange', 'REVIEW': 'yellow', 'MONITOR': 'blue', 'NO ACTION': 'green'}
FILTERS = {'Manoeuvre needed': ['AUTONOMOUS', 'URGENT', 'PLAN'], 'Analyst review': ['REVIEW'], 'Monitoring': ['MONITOR'], 'Safe': ['NO ACTION']}
TIER_NAMES = {'SAFE': 'Safe', 'ELEVATED': 'Elevated', 'CRITICAL': 'Critical'}
TIER_DISPLAY_COLORS = {TIER_NAMES[t]: c for t, c in TIER_COLORS.items()}
MODEL_LABELS = {'LightGBM': 'LightGBM (in use)', 'Neural network (MLP)': 'Neural network', 'Sequence network (GRU)': 'Sequence network',
                'Convolutional network (CNN)': 'Convolutional network', 'LightGBM with PCA': 'LightGBM on compressed features'}
COMPARISON_EVENTS = 6655    # test-like development events scored by model_comparison.py
CHANCE_TICKS = dict(tickvals=[-12, -10, -8, -6, -4, -2], ticktext=['1 in a trillion', '1 in 10 billion', '1 in 100 million', '1 in a million', '1 in 10,000', '1 in 100'])
CLEAN = dict(margin=dict(l=0, r=0, t=36, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
NO_BAR = {'displayModeBar': False}


@st.cache_data
def load():
    need = ['predictions_test.csv', 'config.json', 'final_tier_test.csv', 'final_tier_metrics.csv', 'final_tier_confusion.csv']
    if any(not (OUT / f).exists() for f in need):
        return None
    opt = lambda f, **kw: pd.read_csv(OUT / f, **kw) if (OUT / f).exists() else None
    return dict(p=pd.read_csv(OUT / 'predictions_test.csv'), fin=pd.read_csv(OUT / 'final_tier_test.csv'), cfg=json.load(open(OUT / 'config.json')),
                metrics=pd.read_csv(OUT / 'final_tier_metrics.csv'), confusion=pd.read_csv(OUT / 'final_tier_confusion.csv', index_col=0),
                comparison=opt('final_tier_model_comparison.csv'), w24=opt('warnings_24h_train.csv'), esc=opt('escalation_timing_train.csv'))


@st.cache_data
def simulate(rel, t_sigma, c_sigma, shift):
    return mc_pc(list(rel), t_sigma, c_sigma, shift_t_m=shift, n=2_000_000, seed=1, **D['cfg'].get('mc', {}))


def limit_lines(fig):
    fig.add_hline(y=-4, line_dash='dash', line_color=TIER_COLORS['CRITICAL'], annotation_text='Critical', annotation_position='top left')
    fig.add_hline(y=-6, line_dash='dash', line_color=TIER_COLORS['ELEVATED'], annotation_text='Elevated', annotation_position='top left')


def trajectory_chart(g):
    """Estimated position of the other object at closest approach, one point per update."""
    pos = g[['relative_position_r', 'relative_position_t', 'relative_position_n']].values / 1e3
    reach = max(1.2, 1.15 * np.abs(pos).max())
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    fig = go.Figure()
    for rad, colr, op in [(1, TIER_COLORS['CRITICAL'], .12), (5, TIER_COLORS['ELEVATED'], .05)]:
        if rad <= 2.5 * reach:
            fig.add_trace(go.Surface(x=rad * np.cos(u) * np.sin(v), y=rad * np.sin(u) * np.sin(v), z=rad * np.cos(v), opacity=op,
                                     showscale=False, colorscale=[[0, colr], [1, colr]], hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=pos[:, 1], y=pos[:, 2], z=pos[:, 0], mode='lines+markers', line=dict(color='#495057', width=4),
                               marker=dict(size=5, color='#4c6ef5'), text=[f'{t:.1f} days before' for t in g.time_to_tca],
                               hovertemplate='%{text}<extra></extra>'))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=4, color='black', symbol='diamond'),
                               hovertemplate='Our satellite<extra></extra>'))
    rng = [-reach, reach]
    fig.update_layout(scene=dict(xaxis=dict(title='Along track (km)', range=rng), yaxis=dict(title='Sideways (km)', range=rng),
                                 zaxis=dict(title='Up or down (km)', range=rng), aspectmode='cube'),
                      height=420, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def reset_selection():
    st.session_state.selection_round += 1      # new widget keys clear the selection, so the same event can be reopened


def event_panel(eid):
    g = D['p'][D['p'].event_id == eid].sort_values('time_to_tca', ascending=False)
    last = pd.concat([g.iloc[-1].drop(['action', 'reaction_budget_days']), D['fin'].set_index('event_id').loc[eid]])
    why, steps = recommended_actions(last, g)

    st.badge(ACTION_LABELS[last.action], color=BADGES[last.action])
    st.markdown(f'**{ACTION_TEXT[last.action]}**')
    c = st.columns(3)
    c[0].metric('Forecast final tier', TIER_NAMES[last.forecast_tier])
    c[1].metric('Chance of ending Critical', chance_percent(last.chance_critical))
    if last.forecast_tier == 'CRITICAL':
        c[2].metric('Decide within', duration_text(last.reaction_budget_days))
    else:
        c[2].metric('Closest approach in', duration_text(last.time_to_tca))

    actions, risk, path, trial = st.tabs(['Next actions', 'Risk over time', 'Flight path', 'Try a manoeuvre'])
    with actions:
        st.markdown('\n'.join(f'{i}. {s}' for i, s in enumerate(steps, 1)))
        st.caption('Why: ' + ' '.join(why))
        st.caption(f'Latest update: {last.miss_distance / 1e3:,.2f} km apart, collision chance {chance_text(last.pred_pc)}.')
    with risk:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.time_to_tca, y=g.pred_log10_pc, mode='lines+markers', name='Model estimate', line=dict(color='#4c6ef5', width=2)))
        fig.add_trace(go.Scatter(x=g.time_to_tca, y=g.max_risk_estimate, mode='lines+markers', name='Worst case', line=dict(color='#868e96', width=2, dash='dot')))
        limit_lines(fig)
        fig.update_layout(title='Collision chance', xaxis_title='Days before closest approach', xaxis_autorange='reversed', yaxis=CHANCE_TICKS,
                          height=280, legend=dict(orientation='h', y=1.12, x=1, xanchor='right'), **CLEAN)
        st.plotly_chart(fig, width='stretch', config=NO_BAR)
        fig = go.Figure(go.Scatter(x=g.time_to_tca, y=g.miss_distance / 1e3, mode='lines+markers', line=dict(color='#495057', width=2), name='Miss distance'))
        fig.add_hline(y=1, line_dash='dash', line_color=TIER_COLORS['CRITICAL'])
        fig.add_hline(y=5, line_dash='dash', line_color=TIER_COLORS['ELEVATED'])
        fig.update_layout(title='Miss distance (km)', xaxis_title='Days before closest approach', xaxis_autorange='reversed', height=220, **CLEAN)
        st.plotly_chart(fig, width='stretch', config=NO_BAR)
    with path:
        st.plotly_chart(trajectory_chart(g), width='stretch')
        st.caption('Each point is one update on where the other object will be at closest approach. Spheres mark 1 km and 5 km.')
    with trial:
        need = float(dv_required_mm_s(max(5000 - last.miss_mid, 500), last.time_to_tca))
        a, b = st.columns(2)
        speed = a.slider('Speed change (mm/s)', 0.0, 30.0, min(30.0, np.ceil(need * 4) / 4), 0.25)
        top = float(max(0.5, round(last.time_to_tca, 1)))
        lead = b.slider('Days before closest approach', 0.1, top, top, 0.1)
        rel = (last.relative_position_r, last.relative_position_t, last.relative_position_n)
        shift = float(along_track_shift_m(speed, lead))
        before, low0 = simulate(rel, last.t_sigma_r, last.c_sigma_r, 0.0)
        after, low1 = simulate(rel, last.t_sigma_r, last.c_sigma_r, shift)
        k = st.columns(3)
        k[0].metric('Chance without manoeuvre', ('Under ' if low0 else '') + chance_text(before))
        k[1].metric('Chance after manoeuvre', ('Under ' if low1 else '') + chance_text(after))
        k[2].metric('Added distance', f'{shift / 1e3:.2f} km')
        st.caption('Monte Carlo simulation of 2 million possible positions, for comparing options. Its figures can differ from the model estimate.')


D = load()
if D is None:
    st.error('No results found. Run the notebook first.')
    st.stop()
p = D['p']
st.session_state.setdefault('selection_round', 0)
latest = p.sort_values('time_to_tca').groupby('event_id').head(1).copy()       # most recent message per event
latest = latest.drop(columns=['action', 'reaction_budget_days']).merge(D['fin'], on='event_id')
latest['sev'] = latest.action.map({a: i for i, a in enumerate(ACTIONS)})

st.title('🛰️ Collision Risk Monitor')
st.caption('Forecast by LightGBM. Full details are under Model performance.')
tab_now, tab_ins, tab_model = st.tabs(['Events', 'Insights', 'Model performance'])

# ================================================================== EVENTS
with tab_now:
    c = st.columns(4)
    for col, (name, acts) in zip(c, FILTERS.items()):
        col.metric(name, f'{int(latest.action.isin(acts).sum()):,}')

    choice = st.segmented_control('Show', list(FILTERS), default='Manoeuvre needed', label_visibility='collapsed') or 'Manoeuvre needed'
    q = latest[latest.action.isin(FILTERS[choice])].sort_values(['sev', 'chance_critical'], ascending=[True, False])
    table = pd.DataFrame({'Event': q.event_id.values,
                          'Next step': q.action.map(ACTION_LABELS).values,
                          'Other object': q.c_object_type.map(OBJECT_LABELS).values,
                          'Forecast final tier': q.forecast_tier.map(TIER_NAMES).values,
                          'Chance of ending Critical': q.chance_critical.map(chance_percent).values,
                          'Forecast final distance (km)': (q.miss_mid / 1e3).round(2).values,
                          'Closest approach in (days)': q.time_to_tca.round(1).values})
    st.caption('Forecasts of how each event will end. Test events have no known outcome, so these are demo predictions. Click an event to see what to do next.')
    colors = {ACTION_LABELS[a]: ACTION_COLORS[a] for a in ACTIONS}
    picked = st.dataframe(table.style.format({'Forecast final distance (km)': '{:,.2f}', 'Closest approach in (days)': '{:.1f}'})
                          .apply(lambda s: [f'background-color: {colors[v]}33' for v in s], subset=['Next step']),
                          height=360, width='stretch', hide_index=True, on_select='rerun', selection_mode='single-row',
                          key=f'queue_{choice}_{st.session_state.selection_round}')

    st.subheader('Risk map')
    m = latest.assign(miss_km=latest.miss_mid / 1e3, Tier=latest.forecast_tier.map(TIER_NAMES), chance=latest.pred_pc.map(chance_text),
                      y=latest.pred_log10_pc.clip(lower=-12))
    fig = px.scatter(m, x='miss_km', y='y', color='Tier', color_discrete_map=TIER_DISPLAY_COLORS, log_x=True, opacity=.7, render_mode='svg',
                     category_orders={'Tier': ['Critical', 'Elevated', 'Safe']}, custom_data=['event_id', 'chance'],
                     labels={'miss_km': 'Forecast final distance (km)', 'y': 'Collision chance now'})
    fig.update_traces(marker_size=8, hovertemplate='Event %{customdata[0]}<br>%{customdata[1]}<br>%{x:.1f} km<extra></extra>')
    fig.update_layout(height=380, yaxis=CHANCE_TICKS, xaxis=dict(tickvals=[.1, 1, 5, 10, 50], ticktext=['0.1', '1', '5', '10', '50']), legend=dict(orientation='h', y=1.1, x=1, xanchor='right', title=''), **CLEAN)
    clicked = st.plotly_chart(fig, width='stretch', config=NO_BAR, on_select='rerun', selection_mode='points',
                              key=f'map_{st.session_state.selection_round}')

    eid = None
    if picked.selection.rows:
        eid = int(table.Event.iloc[picked.selection.rows[0]])
    elif clicked.selection.points:
        eid = int(clicked.selection.points[0]['customdata'][0])
    if eid is not None:
        st.dialog(f'Event {eid}', width='large', on_dismiss=reset_selection)(event_panel)(eid)

# ================================================================== INSIGHTS
with tab_ins:
    w24, esc = D['w24'], D['esc']
    if w24 is not None:
        st.subheader('Final 24 hours before closest approach')
        c = st.columns(4)
        c[0].metric('Events', f'{len(w24):,}')
        c[1].metric('Within 5 km', f'{int((w24.miss_distance <= 5000).sum()):,}')
        c[2].metric('Within 1 km', f'{int((w24.miss_distance < 1000).sum()):,}')
        c[3].metric('Chance of 1 in 10,000 or higher', f'{int((w24.risk >= -4).sum()):,}')
        st.caption('From past events in the training data, because the test data stops 2 days before closest approach.')
    if esc is not None:
        st.subheader('Is there time to react?')
        n = len(esc)
        auto = int((esc.first_crit_tca < EXEC_LEAD_DAYS).sum())
        urgent = int(((esc.first_crit_tca >= EXEC_LEAD_DAYS) & (esc.first_crit_tca < EXEC_LEAD_DAYS + HUMAN_MIN_DAYS)).sum())
        crit = latest[latest.forecast_tier == 'CRITICAL']
        gap = (5000 - crit.miss_mid).clip(lower=500)
        now_dv, late_dv = dv_required_mm_s(gap, crit.time_to_tca), dv_required_mm_s(gap, 0.5)
        c = st.columns(3)
        c[0].metric('Time for a person to decide', f'{(n - auto - urgent) / n:.0%}')
        c[1].metric('Urgent decision', f'{urgent / n:.1%}')
        c[2].metric('Too late for a person', f'{auto / n:.1%}')
        l, r = st.columns(2)
        with l:
            fig = px.histogram(esc, x='first_crit_tca', nbins=28, color_discrete_sequence=[TIER_COLORS['CRITICAL']],
                               labels={'first_crit_tca': 'Days before closest approach'})
            fig.update_layout(height=320, xaxis_autorange='reversed', yaxis_title='Events', title='When events first turn critical', bargap=.05, **CLEAN)
            fig.update_traces(hovertemplate='%{x} days: %{y} events<extra></extra>')
            st.plotly_chart(fig, width='stretch', config=NO_BAR)
        with r:
            lead = np.linspace(0.1, 7, 150)
            fig = go.Figure(go.Scatter(x=lead, y=dv_required_mm_s(5000, lead), mode='lines', line=dict(color='#4c6ef5', width=2),
                                       hovertemplate='%{x:.1f} days: %{y:.1f} mm/s<extra></extra>'))
            fig.update_layout(height=320, xaxis_autorange='reversed', yaxis_type='log', yaxis_dtick=1, xaxis_title='Days before closest approach',
                              yaxis_title='Speed change (mm/s)', title='Waiting costs fuel: speed change for 5 km of distance', **CLEAN)
            st.plotly_chart(fig, width='stretch', config=NO_BAR)
        st.markdown(f'For events forecast Critical, acting now needs about **{np.median(now_dv):.1f} mm/s**. '
                    f'Waiting until 12 hours before needs about **{np.median(late_dv / now_dv):.0f} times more**.')

# ================================================================== MODEL
with tab_model:
    st.subheader('Models in use')
    used = st.columns(3)
    for col, (job, model) in zip(used, [('Final distance forecast', 'LightGBM quantile regression'),
                                        ('Chance of ending Critical', 'LightGBM classifier'),
                                        ('Collision chance now', 'Combined model: sequence network, neural network and LightGBM')]):
        col.caption(job)
        col.markdown(f'**{model}**')
    met = D['metrics']
    held = met[met.scope == 'Test-like events'].set_index('system')
    cmp = D['comparison']
    if cmp is not None:
        cmp = cmp[cmp['mode'].isin(['with forecast', 'benchmark'])].copy()
        cmp['name'] = cmp.model.replace(MODEL_LABELS)
        in_use = cmp[cmp.model == 'LightGBM'].iloc[0]
        rival = cmp[(cmp['mode'] == 'with forecast') & (cmp.model != 'LightGBM')].iloc[0]       # file is sorted, best first
        simple = cmp[cmp.model == 'Current tier carried forward'].iloc[0]
        st.subheader('Why LightGBM?')
        st.markdown(f"""
- **Level with the best at catching Critical events:** {in_use.critical_caught:.1%}, against {rival.critical_caught:.1%} for {rival['name'].lower()}. That gap is too small to separate.
- **More of its alerts were right than its closest rival:** {in_use.critical_right:.1%}, against {rival.critical_right:.1%}.
- **Catches more than the simple approach:** {in_use.critical_caught:.1%}, against {simple.critical_caught:.1%}.
""")
    st.subheader('Does the forecast beat the simple approach?')
    order = ['Final system', 'Distance forecast only', 'Current tier carried forward', 'Worst-case figure and distance']
    st.dataframe(pd.DataFrame({'System': order,
                               'Overall score': [f'{held.overall[o]:.1%}' for o in order],
                               'Critical events caught': [f'{held.critical_caught[o]:.1%}' for o in order],
                               'Critical alerts that were right': [f'{held.critical_right[o]:.1%}' for o in order]}),
                 hide_index=True, width='stretch')
    conf = D['confusion']
    st.caption(f'Scored on {int(conf.values.sum()):,} past events held back from training.')
    if cmp is not None:
        final_score = held.critical_caught['Final system']
        critical_held = int(conf.loc['CRITICAL'].sum())
        critical_dev = int(round(in_use.critical_alerts * in_use.critical_right / in_use.critical_caught))
        st.subheader(f'Why does LightGBM show {in_use.critical_caught:.0%} below but {final_score:.0%} above?')
        c = st.columns(2)
        c[0].metric('Model comparison', f'{in_use.critical_caught:.1%}')
        c[0].caption(f'{critical_dev} Critical events across {COMPARISON_EVENTS:,} past events')
        c[1].metric('Final test', f'{final_score:.1%}')
        c[1].caption(f'{critical_held} Critical events across {int(conf.values.sum()):,} past events, kept out of model choices')
        st.markdown(f"These are the same LightGBM system tested on different past events. With {critical_held} Critical events, one event moves the "
                    f"final test by about 1 point, so **{in_use.critical_caught:.0%} to {final_score:.0%}** is the fair way to describe it.")
        st.subheader('All models compared')
        left, right = st.columns([1.2, 1])
        left.dataframe(pd.DataFrame({'Model': cmp.name, 'Critical events caught': cmp.critical_caught.map('{:.1%}'.format),
                                     'Critical alerts that were right': cmp.critical_right.map('{:.1%}'.format),
                                     'Overall score': cmp.overall.map('{:.1%}'.format)}), hide_index=True, width='stretch', height=458)
        most_right = cmp.loc[cmp.critical_right.idxmax()]
        cmp['group'] = np.where(cmp.model == 'LightGBM', 'In use', np.where(cmp.model == most_right.model, 'Most alerts right',
                                np.where(cmp['mode'] == 'benchmark', 'Simple approach', 'Other models')))
        fig = px.scatter(cmp, x='critical_right', y='critical_caught', color='group', custom_data=['name'],
                         color_discrete_map={'In use': '#4c6ef5', 'Most alerts right': '#2a9d8f', 'Simple approach': '#e9a23b', 'Other models': '#adb5bd'},
                         labels={'critical_right': 'Critical alerts that were right', 'critical_caught': 'Critical events caught'})
        fig.update_traces(marker_size=11, hovertemplate='%{customdata[0]}<br>Caught %{y:.1%}, right %{x:.1%}<extra></extra>')
        fig.update_layout(height=458, title='More caught, more false alarms', xaxis_tickformat='.0%', yaxis_tickformat='.0%',
                          legend=dict(orientation='h', y=-0.2, x=0, title=''), **CLEAN)
        right.plotly_chart(fig, width='stretch', config=NO_BAR)
        st.caption('Each model escalates events on top of the distance forecast. Scored on past training events by five-fold testing, '
                   'not on the held-back events. The top models are too close to separate.')
        st.subheader('Which has the most Critical alerts right?')
        added = cmp[cmp['mode'] == 'with forecast'].sort_values('critical_right', ascending=False).iloc[0]
        c = st.columns(2)
        c[0].metric(f"Highest overall: {most_right['name']}", f'{most_right.critical_right:.1%}')
        c[0].caption(f'Catches {most_right.critical_caught:.1%} of Critical events')
        c[1].metric(f"Highest added model: {added['name']}", f'{added.critical_right:.1%}')
        c[1].caption(f'Catches {added.critical_caught:.1%} of Critical events')
        st.markdown(f'These are more often right because they raise fewer alerts, so they catch fewer Critical events than LightGBM ({in_use.critical_caught:.1%}).')
    names = [TIER_NAMES[t] for t in TIERS]
    ct = conf.loc[TIERS, TIERS].set_axis(names).set_axis(names, axis=1)
    fig = px.imshow(ct, text_auto=',', color_continuous_scale='Blues', labels=dict(x='System said', y='Actual', color='Events'))
    fig.update_layout(height=380, title='Final system: right and wrong calls', coloraxis_showscale=False, **CLEAN)
    st.plotly_chart(fig, width='stretch', config=NO_BAR)
