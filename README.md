# LEO-Satellite-Collision-Predictor
Autonomous AI hackathon 2026

Airbus Challenge #1: AI-driven satellite collision avoidance.

The project is in [`collision-risk-evaluator/`](collision-risk-evaluator/). Its README covers results, how it works and how to run it.

## Quick start
```bash
cd collision-risk-evaluator
pip install -r requirements.txt
streamlit run dashboard/app.py
```
The dashboard runs straight away from the saved results in `outputs/`.

## Data
`train_data.csv` and `test_data.csv` are not in the repo (too large for GitHub). To retrain, download them from the hackathon Google Drive (`Challenge #1\archive(1).zip`) and place both files in `collision-risk-evaluator/data/`.
