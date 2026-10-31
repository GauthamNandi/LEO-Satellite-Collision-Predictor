# Collision Risk Monitor: User Guide

The Collision Risk Monitor forecasts how each satellite close approach will end, and suggests what to do next. This guide covers how to run it, how to read each screen, and how to update its results.

**Important:** this is a hackathon prototype. Test events have no known outcome, so the forecasts on screen are demo predictions. Don't use it for real flight decisions.

## Start the dashboard

You need Python and a copy of this repo. The dashboard runs from saved results, so you don't need the large data files.

Run these from the `collision-risk-evaluator` folder.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

**macOS or Linux**
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run dashboard/app.py
```

The first install can take several minutes because it includes PyTorch. Next time, you only need the last command.

The dashboard opens at **http://localhost:8501**. To stop it, press `Ctrl+C` in the terminal.

## Events tab

This is the main screen.

1. **Check the four counts** at the top: Manoeuvre needed, Analyst review, Monitoring and Safe.
2. **Choose a group** with the buttons below the counts. The table shows Manoeuvre needed first.
3. **Click an event** to open its prompt. You can click the box at the start of a table row, or a dot on the risk map.
4. **Close the prompt** with the X or the `Esc` key.

In the table, events are ranked by next step, then by their chance of ending Critical.

On the **risk map**, each dot is one event:
- Further left means a smaller forecast final distance.
- Higher means a higher collision chance now.
- Colour shows the forecast final tier, and the legend names each colour.

## The event prompt

The top of the prompt shows the next step, the forecast final tier and the chance of ending Critical.

For Critical events, it also shows **Decide within**. This is the time left to decide, allowing 12 hours to prepare and carry out the manoeuvre. Other events show the time until closest approach.

| Tab | What it shows |
|---|---|
| **Next actions** | Numbered steps and the reasons for them |
| **Risk over time** | How collision chance and miss distance changed across updates |
| **Flight path** | Where each update placed the other object at closest approach. Drag to rotate it. The spheres mark 1 km and 5 km |
| **Try a manoeuvre** | Choose a speed change and timing, then compare the collision chance before and after |

**Try a manoeuvre** uses a Monte Carlo simulation of 2 million possible positions. Use it to compare options, because it relies on assumed uncertainties. It doesn't change the tier, so its figures can differ from the model estimate.

## What the next steps mean

| Next step | Meaning |
|---|---|
| **Automatic manoeuvre** | Too late for a person to decide. The prepared manoeuvre runs unless an operator stops it |
| **Decide now** | A person must decide soon, with less than a day of decision time left |
| **Plan manoeuvre** | Forecast Critical, with time for a considered decision |
| **Analyst review** | Not forecast Critical, but with at least a 10% chance of ending Critical |
| **Monitor** | Forecast Elevated. Check again at the next update |
| **No action** | Forecast Safe |

You won't see **Automatic manoeuvre** or **Decide now** in the demo, because every test event stops at least 2 days before closest approach.

In a **Plan manoeuvre** prompt, check the first step:
- **"Treat this as a real risk"** means the chance of ending Critical is at least 50%.
- **"Confirm with the next update"** means it's a cautious alert with a lower chance.

## Risk tiers

| Tier | Rule |
|---|---|
| **Critical** | Collision chance of 1 in 10,000 or higher, or miss distance under 1 km |
| **Elevated** | Collision chance from 1 in 1,000,000 to 1 in 10,000, or miss distance from 1 km to 5 km |
| **Safe** | Collision chance below 1 in 1,000,000 and miss distance over 5 km |

## Other tabs

**Insights** shows how many past events came within 1 km and 5 km in the final 24 hours, how much time people had to react, and how waiting raises the speed change needed. These figures come from training data, because test data stops 2 days out.

**Model performance** shows the models in use, the scores on events held back from training, a comparison of all the models, and where the final system was right and wrong.

| Term | Meaning |
|---|---|
| **Critical events caught** | Of the events that really ended Critical, the share the system flagged |
| **Critical alerts that were right** | Of the Critical alerts raised, the share that really ended Critical |
| **Overall score** | A balanced score across all three tiers |

The system deliberately favours catching Critical events, so it raises more false alarms than the simpler options.

## Update the results

You only need this if you change the code or data.

1. Put these in `collision-risk-evaluator/data/`:
   - `train_data.csv` and `test_data.csv`, from the hackathon Google Drive
   - `challenge1_event_labels.xlsx`, which is already in the repo
2. Run these from `collision-risk-evaluator`, using the Python in your `.venv`:
   ```bash
   python -m jupyter nbconvert --to notebook --execute --inplace collision_risk_pipeline.ipynb
   python final_tier_pipeline.py
   python model_comparison.py
   ```
3. Restart the dashboard to load the new results.

| Step | What it updates | Needed for |
|---|---|---|
| Notebook | Collision chance for each message, and the Insights data | Risk over time, collision chance, Insights |
| `final_tier_pipeline.py` | Forecasts and held-back scores | Events tab, Model performance |
| `model_comparison.py` | All-models comparison | Model performance (optional) |

Each step can take several minutes on a laptop.

To change the assumed timings, edit `EXEC_LEAD_DAYS` and `HUMAN_MIN_DAYS` in `risk_utils.py`, then rerun the steps above.

## Troubleshooting

| Problem | What to do |
|---|---|
| "No results found" | Check that `outputs/` holds its files. Run the dashboard from `collision-risk-evaluator` after pulling the latest repo |
| The page doesn't open | Open http://localhost:8501 yourself, and check the terminal for errors |
| Port 8501 is in use | Add `--server.port 8502` to the run command, then open that port |
| Old results still show | Stop and restart the dashboard |
| `final_tier_pipeline.py` can't find a file | Check that all three data files are in `data/`, then run it from `collision-risk-evaluator` |
| Scores look out of date | Rerun the steps in [Update the results](#update-the-results) |

## Limits

- **Misses and false alarms:** on held-back events, the system missed 9 of 101 Critical events, and about a third of its Critical alerts were false alarms.
- **Estimated collision chance:** a true calculation needs full position uncertainty and object sizes, which aren't in the recommended columns.
- **Rough speed changes:** they use a simple drift estimate and assumed timings.
- **Human decision:** analysts keep the final call.
