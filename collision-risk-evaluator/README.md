# Satellite Collision Risk Evaluator
**Autonomous SG 2026 · Airbus Track · Challenge #1: AI-driven satellite collision avoidance**

Reads conjunction data messages (CDMs) and forecasts each event's **final risk tier** from messages at least 2 days before closest approach. It also estimates the collision probability `Pc` of every message, decides **who acts and how urgently** (analyst vs autonomous manoeuvre), suggests next actions, and shows it all on an operations dashboard.

| Tier | Rule (from the brief) |
|---|---|
| **Critical** (manoeuvre) | Pc ≥ 1e-4 **or** miss distance < 1 km |
| **Elevated** | 1e-6 ≤ Pc < 1e-4 **or** miss distance 1–5 km |
| **Safe** | Pc < 1e-6 **and** miss distance > 5 km |

## Results
**Official target:** predict each event's *final* tier using only messages at least 2 days before closest approach (`data/challenge1_event_labels.xlsx`).
Test events have no final outcomes, so scores come from 20 % of training events held out before any modelling, and test output is demo predictions.

Held-out events shaped like the test set (latest message within 1 day of closest approach, 1,638 events):

| System | Overall score (macro-F1) | Critical caught | Critical alerts correct |
|---|---|---|---|
| **Final system** | 0.848 | 91.1% | 64.8% |
| Distance forecast only | 0.874 | 82.2% | 79.8% |
| Current tier carried forward | 0.856 | 84.2% | 73.9% |
| Worst-case figure and distance | 0.703 | 84.2% | 68.0% |

- **The trade-off:** the final system catches more Critical events than carrying the current tier forward, and raises more false alarms. Overall score is not clearly different.
- **Not a measure of skill:** events whose final message is already at least 2 days out are excluded here, because the input would contain the answer.
- **Message-level estimates:** `collision_risk_pipeline.ipynb` still estimates the collision chance of each message. Its scores (macro-F1 0.965) measure that easier task, not the official target.

## Repository

```
collision_risk_pipeline.ipynb   message-level collision chance (EDA → features → PCA → 10 models → export)
final_tier_pipeline.py          official target: forecast each event's final tier, score on held-out events
model_comparison.py             every model family on the official target, development events only
risk_utils.py                   shared logic: tier rules, safety net, urgency levels, Δv physics, Monte Carlo Pc
dashboard/app.py                Streamlit operations dashboard
dashboard/coastlines.json       coastline outlines for the globe, stored locally so it works offline
.streamlit/config.toml          dashboard dark theme
outputs/                        predictions, metrics, figures, config written by the notebook and scripts (read by the dashboard)
USER_GUIDE.md                   how to run and use the dashboard
data/                           put train_data.csv, test_data.csv and challenge1_event_labels.xlsx here (CSV files are not committed)
```

## Run it

```bash
pip install -r requirements.txt
# copy train_data.csv, test_data.csv and challenge1_event_labels.xlsx into data/
jupyter nbconvert --to notebook --execute --inplace collision_risk_pipeline.ipynb   # ~6 min on a laptop CPU
python final_tier_pipeline.py
python model_comparison.py   # optional, takes several minutes
streamlit run dashboard/app.py
```
`outputs/` already contains the saved results, so the dashboard runs straight away without re-training.
See the [User Guide](USER_GUIDE.md) for how to run and use the dashboard.

## How it works

1. **Recommended columns only.** `risk` history did not improve the forecast, so it is used only to score the benchmark. Tested on development events.
2. **Event features.** Latest values plus history from messages at least 2 days out: distance range and trend, worst-case trend, shrinking uncertainty.
3. **Forecast.** LightGBM median quantile regression predicts the final miss distance, which sets the best-guess tier. A class-balanced LightGBM classifier estimates the chance of ending Critical; above a threshold (chosen on development events for the F2 score) the event is escalated.
4. **Honest scoring.** Split by event; 20 % held out before modelling; paired bootstrap against the benchmark of carrying the current tier forward. A risk-history feature set was tried on development events and dropped.
   Every model family was compared on development events with cross-fitted thresholds (`model_comparison.py`). Random forest and LightGBM tied at the top (difference in Critical F2 not distinguishable from zero), so LightGBM stayed and the held-out events were not used for this choice.
5. **Human vs autonomous.** Reaction budget = time to closest approach − 0.5 d to uplink and execute a burn. Critical with ≥ 1 d of budget → **PLAN**; < 1 d → **URGENT**; no budget → **AUTONOMOUS** with operator veto. Non-Critical events with at least a 10 % chance of ending Critical go to **REVIEW**.
6. **Next actions.** Clicking an event gives ordered steps based on the NASA handbook: reduce collision chance by at least 1.5 orders of magnitude, screen the new path, contact active operators, keep an abort.
7. **Cost of waiting.** Along-track separation grows ≈ 3·Δv·t (Clohessy–Wiltshire), so the Δv needed scales as 1/lead time.
8. **Monte Carlo what-if.** Samples relative position at closest approach before and after a simulated burn. A what-if tool, not the classifier.
9. **Globe.** Events are drawn as 5° latitude bands at their typical altitude, because the data has no longitude. Opening an event narrows the globe to that event's own ring. `geocentric_latitude`, `t_h_apo` and `t_h_per` are used for this display only and never reach the models. Coastlines are from [Natural Earth](https://www.naturalearthdata.com/) (public domain).
10. **Encounter replay.** A small embedded player updates only the moving trace, because Plotly's built-in 3D animation resets the camera on every frame. Rotating and zooming therefore keep working during playback.

## Limitations (stated up front)
- Pc is **predicted**, not computed: a true Pc needs full covariances and object sizes, which are outside the recommended columns.
- 895 of 908 final Critical training events (98.6 %) are Critical through miss distance alone, so the forecast is mainly a distance forecast. NASA treats miss distance alone as a weak measure.
- The final system misses some Critical events and raises false alarms; review them rather than relying on the forecast alone.
- Test data stops 2 days before TCA, so the AUTONOMOUS level is demonstrated on the near-TCA training history, not on test.
- The chance of ending Critical is calibrated on training events, but that calibration has not been checked on the held-out events.
- Events are split by event, not by mission, so performance on a new mission is untested.
- It runs on saved data, not live messages.
- Execution lead (0.5 d) and human decision time (1 d) are assumptions; change them in `risk_utils.py`.
- The analyst keeps the final decision except in the autonomous window, where they keep a veto.
