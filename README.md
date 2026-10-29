# SAWT | صوت — Saudi Podcast Intelligence

SAWT is an interactive Streamlit dashboard for exploring Saudi podcast videos on YouTube. The project combines a reproducible EDA notebook, a local dataset, and an optional script for collecting fresh data.

By: Raghad Alharbi.

## Project question

**What seems to be associated with higher views and engagement on Saudi podcast videos?**

## Project files

```text
sawt_dashboard/
├── app.py                 # Streamlit dashboard
├── collect_youtube_data.py  # Optional YouTube Data API collector
├── requirements.txt       # Dashboard dependencies
├── requirements-collector.txt
├── data/
│   └── clean_data.csv     # Dataset used by the dashboard and notebook
└── notebooks/
    └── EDA_Podcast_EDA_Clean_English_improved.ipynb
```

## Run the dashboard

Python 3.9 or newer is recommended.

```bash
cd sawt_dashboard
python -m venv .venv
```

Activate the environment, then install the requirements and run Streamlit:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. The dashboard reads `data/clean_data.csv` locally; it does not need a YouTube API key to run.

On Windows, `python -m streamlit run app.py` is a reliable alternative if `streamlit` is not recognized as a command.

## What is in the dashboard?

- **Overview:** video count, total engagement, and typical video duration and views.
- **Channels:** upload counts, total views, median views, and view distributions.
- **Engagement:** views vs. likes/comments and Spearman correlations.
- **Duration and timing:** duration distribution, duration vs. views, and median views by publication day.
- **Highlights:** most-viewed videos, like-rate leaders, outliers, and channel comparisons.
- **Data explorer:** filtered data, column information, summary statistics, and CSV download.
- **Analysis limits:** notes about what the data can and cannot show.

The sidebar filters by channel, publication day, duration, and year. The sections use the same filtered dataset so their numbers remain consistent.

## Data and analysis rules

The current dataset contains **648 collected videos**. The notebook and dashboard apply these rules for the main analysis:

- Keep videos that are at least **30 minutes** long, leaving **599 videos**.
- Fill missing comment counts with `0`.
- Convert publication timestamps to `Asia/Riyadh` when analyzing publication timing.
- Compare channel-level statistics only for channels with at least **3 videos**.
- Use the IQR rule to flag view-count outliers; outliers are not automatically removed.
- Use Spearman correlation for skewed engagement metrics.
- Calculate like rate as likes divided by views, and apply a median-views cutoff when ranking videos by like rate.

A few very popular videos have a large effect on total views and averages. They remain in the analysis because they are real observations, but this limitation is noted in the dashboard.

## Optional: collect fresh YouTube data

The collector is separate from the notebook and dashboard. It requires a YouTube Data API key.

```bash
pip install -r requirements-collector.txt
```

Set the key as an environment variable:

```bash
# macOS / Linux
export YOUTUBE_API_KEY="your-key"

# Windows PowerShell
$env:YOUTUBE_API_KEY="your-key"
```

Then run:

```bash
python collect_youtube_data.py --output data/saudi_podcast_videos.csv --clean-output data/clean_data_new.csv
```

The script does not overwrite existing files unless `--overwrite` is passed. Each YouTube search page uses 100 quota units, and the default daily quota is usually 10,000 units.

To use another cleaned CSV with the dashboard:

```bash
# Windows PowerShell
$env:SAWT_DATA_FILE="data/clean_data_new.csv"
python -m streamlit run app.py
```

```bash
# macOS / Linux
SAWT_DATA_FILE=data/clean_data_new.csv python -m streamlit run app.py
```

The CSV needs video ID, title, channel, publication date, views, likes, comments, and a duration column (`duration_minutes` or ISO 8601 `duration`).

## Notes and limitations

This is descriptive EDA, not a prediction model. Correlation does not prove causation. The dataset comes from keyword-based YouTube searches, so it is not a complete list of Saudi podcasts. Older videos have also had more time to collect views.
