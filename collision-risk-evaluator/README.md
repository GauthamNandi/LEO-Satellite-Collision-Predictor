# Satellite Collision Risk Evaluator
**Autonomous SG 2026 · Airbus Track · Challenge #1: AI-driven satellite collision avoidance**

Reads conjunction data messages (CDMs), predicts the collision probability `Pc` for every message, assigns the NASA/ESA risk tier from the brief, decides **who acts and how urgently** (analyst vs autonomous manoeuvre), and shows it all on an operations dashboard.

| Tier | Rule (from the brief) |
|---|---|
| **Critical** (manoeuvre) | Pc ≥ 1e-4 **or** miss distance < 1 km |
| **Elevated** | 1e-6 ≤ Pc < 1e-4 **or** miss distance 1–5 km |
| **Safe** | Pc < 1e-6 **and** miss distance > 5 km |

## Results (test set: 2,167 unseen events, 24,484 CDMs)

| | Baseline (`max_risk_estimate` as Pc) | Best ML alone (ensemble: GRU + MLP + LightGBM) | **Final system (ML + safety net)** |
|---|---|---|---|
| Macro-F1 (3 tiers) | 0.841 | 0.977 | **0.965** |
| Critical recall | 1.000 | 0.982 | **1.000** |
| Critical precision | 0.928 | 0.996 | 0.928 |
| Pc-critical CDMs caught | 100 % | 83 % | **100 %** |
| Safe CDMs wrongly raised to Elevated | 29 % | ~1 % | ~1 % |

- **No missed manoeuvres:** every Critical CDM in the test set is flagged, and the false-Elevated rate drops from 29 % to about 1 %.
- **Early warning:** the first Critical flag comes a median **6.1 days** before closest approach (minimum 2.1 d), giving an analyst about **5.6 days** to decide.
- **When autonomy is needed:** in the full history, 93 % of events that turn Critical do so with ≥ 1.5 days to go; **1.7 %** turn Critical inside 12 h, too late for a human.
- **Cost of waiting:** acting at the first flag needs a median **2.5 mm/s** Δv to open 5 km of separation; waiting until 12 h before closest approach needs about **12×** more.
- **Explored and rejected:** a selective safety net cut Critical alerts by 3–4 % but missed 2 Critical CDMs on test, so it is not used.

## Repository

```
collision_risk_pipeline.ipynb   full pipeline, executed (EDA → features → PCA → 10 models → evaluation → insights → export)
risk_utils.py                   shared logic: tier rules, safety net, urgency levels, Δv physics, Monte Carlo Pc
dashboard/app.py                Streamlit operations dashboard
outputs/                        predictions, metrics, figures, config written by the notebook (read by the dashboard)
data/                           put train_data.csv and test_data.csv here (not committed)
```

## Run it

```bash
pip install -r requirements.txt
# copy train_data.csv and test_data.csv into data/
jupyter nbconvert --to notebook --execute --inplace collision_risk_pipeline.ipynb   # ~6 min on a laptop CPU
streamlit run dashboard/app.py
```
`outputs/` already contains the results of the executed notebook, so the dashboard runs straight away without re-training.

## How it works

1. **Allowed inputs only.** The 13 columns listed in the brief; `risk` (= log10 Pc) is used only as the label.
2. **Match the test conditions.** Every test CDM is ≥ 2 days before closest approach (TCA), so training uses the same window. Test *inputs* were inspected (ranges, drift via KS test); test *labels* are never used for choices.
3. **Features.** Physics features (log miss distance, miss distance in units of combined radial sigma, …) plus causal per-event history (trend of `max_risk_estimate`, change in miss distance). Pearson-correlation filter, drift filter, redundancy filter → 21 features → PCA keeping 99 % of variance.
4. **Models.** Elliptic Envelope (anomaly detection), Ridge, Random Forest, XGBoost, LightGBM (with/without PCA), LightGBM classifier, MLP, GRU (RNN) and causal 1D-CNN over each event's CDM sequence, and a top-3 ensemble. Selected on a validation split grouped by event; test scored once.
5. **Tiering + safety net.** The ML predicts log10 Pc; tiers come from the brief's rules. Any CDM whose own upper-bound Pc (`max_risk_estimate`) is ≥ 1e-4 is escalated to Critical, so no Pc-critical encounter is missed.
6. **Human vs autonomous.** Reaction budget = time to TCA − 0.5 d to uplink and execute a burn. Critical with ≥ 1 d of budget → **PLAN** (analyst); < 1 d → **URGENT** (human now, plan armed); no budget and a confident model → **AUTONOMOUS** (execute with operator veto). Uncertain predictions near a threshold go to an analyst **REVIEW** queue.
7. **Cost of waiting.** Along-track separation grows ≈ 3·Δv·t (Clohessy–Wiltshire), so the Δv needed scales as 1/lead time.
8. **Monte Carlo what-if.** Samples the relative position at TCA and counts hits on the hard-body sphere, before and after a simulated burn. Unknown along-track/normal sigmas and hard-body radius are calibrated on training data. It is a what-if tool, not the classifier.

## Limitations (stated up front)
- Pc is **predicted**, not computed: a true Pc needs full covariances and object sizes, which are outside the allowed columns.
- About 95 % of Critical CDMs are Critical because miss distance < 1 km, which the rule handles exactly; the *Pc-tier* metrics isolate what the ML contributes.
- Test data stops 2 days before TCA, so the AUTONOMOUS level is demonstrated on the near-TCA training history, not on test.
- Execution lead (0.5 d) and human decision time (1 d) are assumptions; change them in `risk_utils.py`.
- The analyst keeps the final decision except in the autonomous window, where they keep a veto.
