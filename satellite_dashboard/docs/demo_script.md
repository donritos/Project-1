# Three-minute demo

Before you start, run `uv run python scripts/validate_bundle.py` (everything should say OK), then
`uv run streamlit run app.py`. Leave the sidebar on **Held-out training events**.

**0:00 — The task and the cutoff (Overview)**
- "Operators get thousands of conjunction messages. We forecast each event's *final* tier, the tier of its last
  message before closest approach, using only messages at least two days out." Point at the eligibility rule in the
  sidebar.
- KPI cards: 2,389 held-out events, 224 forecast to finish Critical. Point at the tier-movement matrix: forecasts
  off the diagonal are events the model expects to escalate or de-escalate.

**0:40 — One forecast-Critical event (Event detail)**
- Type `1463` in *Search event ID* and open it.
- Read the three panels left to right: **available evidence** (latest eligible CDM 2.33 days out, observed miss
  distance 1.17 km, latest observed tier only *Elevated*), the **forecast** (*Critical*), and the **historical
  outcome** (*Critical, evaluation only*).
- "Carrying the latest tier forward would have said Elevated. The model said Critical, two days ahead."

**1:20 — How the evidence evolved (Forecast history tab)**
- P(final = Critical) is about 4% while the miss distance sits near 10 km. It climbs to about 90% around 3 days out
  as the miss distance falls below 1 km, then eases to 51% at the last eligible message when the miss distance widens
  back to 1.17 km. Each point uses only the messages received up to then. The model keeps the event Critical even
  though the latest observed tier is only Elevated.
- Point out the measurement panel underneath, and the probability bar: "this is the probability of the final *tier*,
  not a collision probability, and the app says when an uncertainty interval is unavailable."
- Optional: open **Explanation** for the drivers, then **Review & export**, save an "Acknowledged" note, and show that
  the event is still counted as Critical.

**2:00 — An eligible upload (Upload & predict)**
- Click **Try the sample file**: 64 rows, 24 of them inside the 2-day cutoff and excluded before the model runs.
  Event 5 has no eligible message, so it shows *Insufficient eligible data*, never Safe.

**2:30 — Held-out performance and an honest failure (Model evidence)**
- Real-forecast events (1,779): Critical recall 0.907 against 0.797 for the carry-forward baseline, Critical F2
  0.848 against 0.790. 18 of 24 escalations are flagged (13 forecast Critical, 5 on watch).
- Say the trade-off out loud: precision is lower (0.673 against 0.764). The model accepts more false alarms in
  exchange for fewer missed Critical events.
- Open **Failure cases → Missed Critical**, event **2958**: one eligible message, 34.5 km away, P(Critical) ≈ 0, yet
  it ended Critical. "A sudden late jump with no warning in the eligible data. ESA's 2019 competition found the
  same hard cases. This is why the app keeps an analyst in the loop."
