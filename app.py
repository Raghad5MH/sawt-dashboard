# -*- coding: utf-8 -*-
"""
SAWT | صوت — Saudi Podcast Intelligence
=======================================

Interactive Streamlit dashboard for the exploratory analysis of Saudi podcast
videos on YouTube. It reuses the analytical decisions of the project notebook
(notebooks/EDA_Podcast_EDA_Clean_English_improved.ipynb):

  * cleaned data file ........ data/clean_data.csv
  * duration rule ............ keep videos of 30 minutes or more
  * comments ................. missing values filled with 0
  * timestamps ............... converted to Asia/Riyadh
  * channel statistics ....... channels with at least 3 videos (adjustable)
  * outliers ................. IQR rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR) on views
  * correlations ............. Spearman (rank based, robust to viral videos)
  * like rate ................ likes / views, ranked among videos with at least
                               the median number of views

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SAWT | صوت — Saudi Podcast Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. Constants
# =============================================================================
APP_DIR = Path(__file__).resolve().parent


def find_data_file() -> Path:
    """SAWT_DATA_FILE, then data/clean_data.csv, then clean_data.csv next to app.py."""
    env = os.environ.get("SAWT_DATA_FILE")
    candidates = ([Path(env)] if env else []) + [APP_DIR / "data" / "clean_data.csv", APP_DIR / "clean_data.csv"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_FILE = find_data_file()

MIN_DURATION_MINUTES = 30  # notebook rule: keep videos >= 30 minutes
DEFAULT_MIN_VIDEOS = 3     # notebook rule: channels with >= 3 videos
MIN_DAY_SAMPLE = 5         # a publication day needs >= 5 videos to be compared
MIN_CORR_SAMPLE = 8        # minimum rows to report a correlation
TOP_N_CHANNELS = 12

C = {
    "obsidian": "#101113",
    "graphite": "#1B1D20",
    "ivory": "#F2EBDD",
    "bronze": "#B69A70",
    "amber": "#E7A94B",
    "muted": "#92949A",
}
PLOT_FONT = "IBM Plex Sans Arabic, Noto Sans Arabic, Segoe UI, Tahoma, sans-serif"
GRID = "rgba(242,235,221,0.07)"
AXIS_LINE = "rgba(242,235,221,0.18)"
CORR_SCALE = [[0.0, "#5F7589"], [0.5, "#1B1D20"], [1.0, "#E7A94B"]]

# pandas dayofweek: Monday=0 ... Sunday=6. Saudi week order: Sunday first.
DAY_AR = {0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس", 4: "الجمعة", 5: "السبت", 6: "الأحد"}
DAY_ORDER = [6, 0, 1, 2, 3, 4, 5]

NAV = [
    ("overview", "نظرة عامة"),
    ("channels", "القنوات"),
    ("engagement", "التفاعل"),
    ("timing", "المدة والنشر"),
    ("signals", "لحظات لافتة"),
    ("explorer", "البيانات"),
    ("limits", "حدود التحليل"),
]

COLUMN_DOCS = {
    "video_id": "معرّف الفيديو على يوتيوب",
    "video_title": "عنوان المقطع",
    "channel_name": "اسم القناة",
    "channel_id": "معرّف القناة",
    "published_at": "تاريخ النشر (بتوقيت الرياض)",
    "matched_search": "عبارة البحث التي التقطت المقطع",
    "views": "عدد المشاهدات",
    "likes": "عدد الإعجابات",
    "comments": "عدد التعليقات (القيم المفقودة = 0)",
    "duration": "المدة بصيغة ISO 8601",
    "duration_minutes": "المدة بالدقائق",
    "video_url": "رابط المقطع",
    "collected_at": "وقت جمع البيانات (UTC)",
    "publish_dow": "رقم يوم النشر (الاثنين = 0)",
    "publish_year": "سنة النشر",
    "publish_hour": "ساعة النشر (توقيت الرياض)",
    "video_link": "رابط مُتحقَّق منه (من معرّف صالح فقط)",
}

# =============================================================================
# 2. Style (CSS)
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--obsidian:#101113;--graphite:#1B1D20;--ivory:#F2EBDD;--bronze:#B69A70;--amber:#E7A94B;--muted:#92949A;
--line:rgba(182,154,112,.24);--line-soft:rgba(242,235,221,.09);
--ar:'IBM Plex Sans Arabic','Noto Sans Arabic','Segoe UI',Tahoma,sans-serif;
--en:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;
--mono:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace}
html,body,[data-testid="stAppViewContainer"]{background:var(--obsidian)!important;color:var(--ivory);font-family:var(--ar)}
.stApp{background:radial-gradient(1100px 520px at 88% -8%,rgba(231,169,75,.10),transparent 60%),radial-gradient(900px 480px at 0% 6%,rgba(182,154,112,.07),transparent 55%),var(--obsidian)!important}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1240px;padding:2.2rem 2rem 5rem}
@media (max-width:640px){.block-container,[data-testid="stMainBlockContainer"]{padding:1rem 1rem 4rem}}
[data-testid="stHeader"]{background:transparent}
#MainMenu,footer,[data-testid="stDecoration"],.stDeployButton{visibility:hidden;display:none}
[data-testid="stSidebar"]{background:var(--graphite);border-inline-end:1px solid var(--line)}
[data-testid="stWidgetLabel"]{direction:rtl;text-align:right;width:100%}
[data-testid="stWidgetLabel"] p,[data-testid="stSidebar"] label p{font-family:var(--ar);color:var(--ivory);font-weight:500;font-size:.92rem}
span[data-baseweb="tag"]{background:rgba(231,169,75,.16)!important;color:var(--ivory)!important;border:1px solid rgba(231,169,75,.5);border-radius:8px}
.stButton>button{width:100%;background:transparent;border:1px solid var(--bronze);color:var(--ivory);border-radius:10px;font-family:var(--ar);transition:border-color .2s,color .2s,transform .2s}
.stButton>button:hover{border-color:var(--amber);color:var(--amber);transform:translateY(-1px)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(27,29,32,.55)}
[data-testid="stExpander"] summary p{font-family:var(--ar);font-weight:600}
button[data-baseweb="tab"]{font-family:var(--ar)}
a:focus-visible,button:focus-visible,[role="slider"]:focus-visible,summary:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
.sawt{direction:rtl;text-align:right;font-family:var(--ar)}
.sawt a{color:var(--ivory)}
@keyframes sawt-rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes sawt-pulse{0%,100%{transform:scaleY(1)}50%{transform:scaleY(.62)}}
.sawt-logo{display:inline-flex;align-items:center;gap:12px;direction:ltr}
.logo-bars{display:flex;align-items:center;gap:3px;height:34px}
.logo-bars i{width:4px;height:calc(var(--h)*100%);border-radius:3px;background:var(--amber);animation:sawt-pulse 2.4s ease-in-out infinite}
.logo-bars i:nth-child(2n){animation-delay:-.6s}.logo-bars i:nth-child(3n){animation-delay:-1.2s}
.logo-word{display:flex;align-items:center;gap:10px}
.logo-word .en{font-family:var(--en);font-weight:600;letter-spacing:.32em;font-size:1.25rem;color:var(--ivory)}
.logo-word .sep{width:1px;height:22px;background:var(--bronze)}
.logo-word .ar{font-family:var(--ar);font-weight:700;font-size:1.35rem;color:var(--amber)}
.sawt-logo.compact .logo-bars{height:26px}.sawt-logo.compact .en{font-size:1rem}.sawt-logo.compact .ar{font-size:1.1rem}
.sawt-hero{position:relative;overflow:hidden;isolation:isolate;border:1px solid var(--line);border-radius:22px;padding:clamp(22px,4.5vw,52px);background:linear-gradient(160deg,rgba(27,29,32,.97),rgba(16,17,19,.97));box-shadow:0 30px 80px -40px rgba(0,0,0,.9),inset 0 1px 0 rgba(242,235,221,.05);animation:sawt-rise .7s ease both}
.sawt-hero::before{content:"";position:absolute;inset:0;z-index:-1;background:radial-gradient(520px 260px at 12% 0%,rgba(231,169,75,.16),transparent 70%),repeating-linear-gradient(90deg,rgba(242,235,221,.035) 0 1px,transparent 1px 64px)}
.hero-top{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:clamp(20px,4vw,44px)}
.hero-badge{font-family:var(--en);font-size:.78rem;letter-spacing:.22em;text-transform:uppercase;color:var(--bronze);border:1px solid var(--line);border-radius:999px;padding:6px 14px}
.hero-eyebrow{font-size:.9rem;color:var(--bronze);margin-bottom:10px}
.hero-h1{font-size:clamp(2.2rem,6vw,4.4rem);font-weight:700;line-height:1.18;color:var(--ivory);margin:0 0 14px}
.hero-h1 em{font-style:normal;color:var(--amber)}
.hero-p{max-width:62ch;color:rgba(242,235,221,.82);font-size:1.05rem;line-height:2;margin:0 0 10px}
.hero-q{max-width:62ch;color:var(--bronze);font-size:.98rem;line-height:1.9;margin:0 0 22px;padding-inline-start:14px;border-inline-start:2px solid var(--amber)}
.hero-cta{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.sawt .btn,.sawt-hero a.btn{display:inline-flex;align-items:center;gap:10px;padding:12px 24px;border-radius:999px;background:var(--amber);color:#101113!important;font-weight:600;text-decoration:none!important;transition:transform .2s,box-shadow .2s}
.sawt-hero a.btn:hover{transform:translateY(-2px);box-shadow:0 14px 30px -12px rgba(231,169,75,.7)}
.hero-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-top:clamp(22px,3.5vw,38px)}
.hero-stats .stat{border-inline-start:2px solid var(--amber);padding:2px 14px}
.hero-stats .stat b{display:block;font-family:var(--mono);font-weight:500;font-size:1.65rem;color:var(--ivory);direction:ltr;text-align:right}
.hero-stats .stat span{color:var(--muted);font-size:.85rem}
.wave{display:flex;align-items:center;gap:3px;height:92px;margin-top:clamp(20px,3vw,32px);direction:ltr}
.wave i{flex:1;min-width:2px;height:calc(var(--h)*100%);border-radius:3px;background:linear-gradient(180deg,var(--amber),var(--bronze));opacity:.88;transform-origin:center;animation:sawt-pulse 2.8s ease-in-out infinite;animation-delay:calc(var(--i)*-70ms)}
.wave-cap{margin-top:8px;color:var(--muted);font-size:.78rem}
.wave-div{display:flex;align-items:center;gap:3px;height:16px;margin:14px 0 4px;opacity:.55;direction:ltr}
.wave-div i{width:3px;height:calc(var(--h)*100%);border-radius:2px;background:var(--bronze)}
.sawt-nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px}
.sawt-nav a{padding:7px 14px;border:1px solid var(--line-soft);border-radius:999px;color:var(--muted)!important;font-size:.88rem;text-decoration:none!important;transition:color .2s,border-color .2s,background .2s}
.sawt-nav a:hover{color:var(--amber)!important;border-color:var(--amber);background:rgba(231,169,75,.06)}
.sec{margin:clamp(40px,7vw,84px) 0 18px;scroll-margin-top:70px}
.sec-kicker{display:flex;align-items:center;gap:12px;font-family:var(--mono);font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:var(--bronze);direction:rtl}
.sec-num{color:var(--amber);font-size:.95rem}
.sec-title{font-size:clamp(1.8rem,4vw,2.9rem);font-weight:700;line-height:1.25;margin:.35rem 0 .5rem;color:var(--ivory)}
.sec-lead{color:var(--muted);max-width:72ch;font-size:1rem;line-height:1.95;margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.kpi{background:linear-gradient(180deg,rgba(27,29,32,.96),rgba(22,23,26,.96));border:1px solid var(--line-soft);border-radius:16px;padding:16px 18px 14px;transition:transform .25s,border-color .25s,box-shadow .25s;animation:sawt-rise .6s ease both;animation-delay:calc(var(--k,0)*60ms)}
.kpi:hover{transform:translateY(-3px);border-color:var(--amber);box-shadow:0 18px 40px -24px rgba(231,169,75,.5)}
.kpi-tag{display:inline-block;font-size:.72rem;padding:2px 10px;border-radius:999px;border:1px solid var(--line);color:var(--bronze);margin-bottom:8px}
.kpi-val{font-family:var(--mono);font-size:clamp(1.7rem,3vw,2.3rem);font-weight:500;color:var(--ivory);direction:ltr;text-align:right;line-height:1.15}
.kpi-label{font-size:.98rem;margin-top:6px;color:var(--ivory)}
.kpi-sub{font-size:.78rem;color:var(--muted);font-family:var(--en);direction:ltr;text-align:right}
.insight{margin:16px 0 6px;padding:16px 20px;border:1px solid var(--line-soft);border-inline-start:3px solid var(--amber);border-radius:12px;background:rgba(27,29,32,.72)}
.insight.warn{border-inline-start-color:var(--bronze)}
.insight-h{font-size:.82rem;letter-spacing:.03em;color:var(--amber);font-weight:600;margin-bottom:6px}
.insight-b{color:var(--ivory);line-height:2;font-size:.98rem}
.insight-b b{color:var(--amber);font-weight:600}
.chart-h{margin:22px 0 2px}
.chart-t{font-weight:600;font-size:1.05rem;color:var(--ivory)}
.chart-s{color:var(--muted);font-size:.82rem;font-family:var(--en);direction:ltr;text-align:right}
.note{color:var(--muted);font-size:.86rem;line-height:1.8;padding:10px 0}
.rank{display:flex;flex-direction:column;gap:8px;margin-top:8px}
.rank-row{position:relative;display:grid;grid-template-columns:34px minmax(0,1fr) auto;gap:12px;align-items:center;padding:10px 14px;border:1px solid var(--line-soft);border-radius:12px;overflow:hidden;background:rgba(27,29,32,.55)}
.rank-row::before{content:"";position:absolute;inset-block:0;inset-inline-start:0;width:var(--w);background:linear-gradient(270deg,rgba(231,169,75,.24),rgba(231,169,75,.03))}
.rank-row>div{position:relative}
.rank-n{font-family:var(--mono);color:var(--bronze);font-size:.9rem}
.rank-t{min-width:0}
.rank-t a,.rank-t span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ivory)!important;text-decoration:none!important;font-weight:500}
.rank-t a:hover{color:var(--amber)!important;text-decoration:underline!important}
.rank-m{color:var(--muted);font-size:.8rem;margin-top:2px}
.rank-v{font-family:var(--mono);color:var(--amber);direction:ltr;text-align:left;white-space:nowrap;font-size:.95rem}
.limits{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.limit{border:1px solid var(--line-soft);border-radius:14px;padding:14px 18px;background:rgba(27,29,32,.6);line-height:1.9;font-size:.95rem}
.limit b{color:var(--amber)}
.limit .ln{font-family:var(--mono);color:var(--bronze);font-size:.8rem;display:block;margin-bottom:2px}
.empty{border:1px dashed var(--line);border-radius:16px;padding:36px 24px;text-align:center;color:var(--muted);line-height:2;margin-top:24px}
.empty b{color:var(--ivory);display:block;font-size:1.2rem;margin-bottom:6px}
.side-h{font-size:.8rem;letter-spacing:.12em;color:var(--bronze);margin:18px 0 4px;font-family:var(--mono);text-transform:uppercase}
.side-p{color:rgba(242,235,221,.78);font-size:.88rem;line-height:1.9}
.side-status{margin-top:14px;padding:10px 14px;border:1px solid var(--line);border-radius:12px;font-size:.88rem;line-height:1.8;color:var(--ivory)}
.side-status b{color:var(--amber);font-family:var(--mono)}
.foot{margin-top:70px;padding-top:18px;border-top:1px solid var(--line-soft);color:var(--muted);font-size:.82rem;line-height:1.9}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
"""


def inject_css() -> None:
    st.markdown("<style>" + re.sub(r"\s+", " ", CSS) + "</style>", unsafe_allow_html=True)


# =============================================================================
# 3. Small helpers
# =============================================================================
def squash(snippet: str) -> str:
    """Collapse an HTML snippet to one line (Markdown treats indented lines as code)."""
    return re.sub(r">\s+<", "><", " ".join(snippet.split()))


def esc(text) -> str:
    """Escape untrusted text for HTML and keep Streamlit from reading '$' as LaTeX."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def pesc(text) -> str:
    """Escape text for Plotly labels and hover boxes."""
    return html.escape(str(text), quote=False)


def trunc(text, n: int = 70) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def fmt_int(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{int(round(float(v))):,}"


def fmt_compact(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= limit:
            return f"{v / limit:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{v:,.0f}"


def pct(x, digits: int = 0) -> str:
    return f"{x:.{digits}f}%"


_ISO = re.compile(r"^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")


def iso_to_minutes(value) -> float:
    """ISO 8601 duration (e.g. PT3H3M9S) -> minutes."""
    if not isinstance(value, str):
        return np.nan
    m = _ISO.match(value.strip())
    if not m or not any(m.groups()):
        return np.nan
    dd, hh, mm, ss = m.groups()
    return int(dd or 0) * 1440 + int(hh or 0) * 60 + int(mm or 0) + float(ss or 0) / 60


def spearman(x: pd.Series, y: pd.Series):
    """Spearman rho = Pearson correlation of the ranks. Returns (rho | None, n)."""
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < MIN_CORR_SAMPLE:
        return None, n
    rx, ry = pair.iloc[:, 0].rank(), pair.iloc[:, 1].rank()
    if rx.nunique() < 2 or ry.nunique() < 2:
        return None, n
    return float(rx.corr(ry)), n


def strength_ar(rho: float) -> str:
    a = abs(rho)
    if a >= 0.7:
        label = "قوي"
    elif a >= 0.4:
        label = "متوسط"
    elif a >= 0.2:
        label = "ضعيف"
    else:
        return "شبه معدوم"
    return f"{label} {'موجب' if rho > 0 else 'سالب'}"


def iqr_fences(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def binned_median(x: pd.Series, y: pd.Series, bins: int = 8):
    """Median of y inside quantile bins of x: a robust trend line for skewed data."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    d = d[(d["x"] > 0) & (d["y"] > 0)]
    if len(d) < 40 or d["x"].nunique() < 4:
        return None
    d["bin"] = pd.qcut(d["x"], q=min(bins, d["x"].nunique()), duplicates="drop")
    g = d.groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "median"))
    return g if len(g) >= 3 else None


# =============================================================================
# 4. Data loading and cleaning (mirrors the notebook)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(path: str, mtime: float):
    """Load the cleaned dataset and apply the notebook rules. Returns (df, meta)."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = ["video_id", "video_title", "channel_name", "published_at", "views", "likes", "comments"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("أعمدة مطلوبة غير موجودة في ملف البيانات: " + ", ".join(missing))

    meta = {"raw_rows": int(len(raw))}
    df = raw.drop_duplicates(subset="video_id").copy()
    meta["duplicates_removed"] = meta["raw_rows"] - int(len(df))

    for col in ("views", "likes", "comments"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["comments"] = df["comments"].fillna(0)  # notebook decision

    if "duration_minutes" in df.columns:
        minutes = pd.to_numeric(df["duration_minutes"], errors="coerce")
    else:
        minutes = pd.Series(np.nan, index=df.index)
    if "duration" in df.columns:
        minutes = minutes.fillna(df["duration"].map(iso_to_minutes))
    df["duration_minutes"] = minutes

    stamp = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_convert("Asia/Riyadh")
    df["published_at"] = stamp

    df["channel_name"] = df["channel_name"].fillna("غير معروف").astype(str)
    df["video_title"] = df["video_title"].fillna("").astype(str)

    n0 = len(df)
    df = df[df["published_at"].notna() & df["views"].notna()].copy()
    meta["bad_rows_removed"] = n0 - int(len(df))

    n1 = len(df)
    df = df[df["duration_minutes"].notna() & (df["duration_minutes"] >= MIN_DURATION_MINUTES)].copy()
    meta["short_removed"] = n1 - int(len(df))

    df["publish_dow"] = df["published_at"].dt.dayofweek.astype(int)
    df["publish_year"] = df["published_at"].dt.year.astype(int)
    df["publish_hour"] = df["published_at"].dt.hour.astype(int)

    # Link only when the video id is a valid 11-character YouTube id.
    ok = df["video_id"].astype(str).str.fullmatch(r"[A-Za-z0-9_-]{11}")
    df["video_link"] = np.where(ok, "https://www.youtube.com/watch?v=" + df["video_id"].astype(str), "")

    df["title_hover"] = df["video_title"].map(lambda t: pesc(trunc(t, 70)))
    df["channel_hover"] = df["channel_name"].map(pesc)

    meta["kept"] = int(len(df))
    collected = None
    if "collected_at" in df.columns:
        ts = pd.to_datetime(df["collected_at"], errors="coerce", utc=True).max()
        if pd.notna(ts):
            collected = ts.strftime("%Y-%m-%d")
    meta["collected_on"] = collected
    return df.reset_index(drop=True), meta


# =============================================================================
# 5. HTML building blocks
# =============================================================================
def logo_html(compact: bool = False) -> str:
    heights = [0.35, 0.6, 0.9, 0.5, 1.0, 0.7, 0.4, 0.8, 0.45]
    bars = "".join(f'<i style="--h:{x}"></i>' for x in heights)
    cls = "sawt-logo compact" if compact else "sawt-logo"
    return squash(
        f'<div class="{cls}"><div class="logo-bars" aria-hidden="true">{bars}</div>'
        '<div class="logo-word"><span class="en">SAWT</span><span class="sep"></span><span class="ar">صوت</span></div></div>'
    )


def wave_html(heights, cls: str = "wave") -> str:
    bars = "".join(f'<i style="--h:{hh:.3f};--i:{i}"></i>' for i, hh in enumerate(heights))
    return f'<div class="{cls}" aria-hidden="true">{bars}</div>'


def divider_html() -> str:
    heights = [0.25 + 0.75 * abs(math.sin(i * 0.55) * math.cos(i * 0.21)) for i in range(64)]
    return wave_html(heights, "wave-div")


def data_wave(df: pd.DataFrame, bars: int = 72):
    """Waveform whose bar height is the median log10(views) of consecutive videos (by date)."""
    s = df.sort_values("published_at")["views"].to_numpy(dtype=float)
    if len(s) == 0:
        return [0.5] * 24
    s = np.log10(np.clip(s, 1, None))
    bars = max(8, min(bars, len(s)))
    med = np.array([np.median(c) for c in np.array_split(s, bars) if len(c)])
    lo, hi = float(med.min()), float(med.max())
    if hi - lo < 1e-9:
        return [0.5] * len(med)
    return list(0.18 + 0.82 * (med - lo) / (hi - lo))


def section_header(num: str, title: str, en: str, lead: str, anchor: str) -> None:
    st.markdown(
        squash(
            f'<div class="sawt sec" id="{anchor}"><div class="sec-kicker"><span class="sec-num">{num}</span>'
            f'<span>{esc(en)}</span></div><div class="sec-title" role="heading" aria-level="2">{esc(title)}</div>'
            f'<p class="sec-lead">{esc(lead)}</p>{divider_html()}</div>'
        ),
        unsafe_allow_html=True,
    )


def chart_head(title: str, sub: str = "") -> None:
    sub_html = f'<div class="chart-s">{esc(sub)}</div>' if sub else ""
    st.markdown(
        squash(f'<div class="sawt chart-h"><div class="chart-t">{esc(title)}</div>{sub_html}</div>'),
        unsafe_allow_html=True,
    )


def insight_card(label: str, body_html: str, warn: bool = False) -> None:
    cls = "sawt insight warn" if warn else "sawt insight"
    st.markdown(
        squash(f'<div class="{cls}"><div class="insight-h">{esc(label)}</div><div class="insight-b">{body_html}</div></div>'),
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="sawt note">{esc(text)}</div>', unsafe_allow_html=True)


def kpi_card(i: int, tag: str, label: str, value: str, sub: str, full: str = "") -> str:
    title = f' title="{esc(full)}"' if full else ""
    return (
        f'<div class="kpi" style="--k:{i}"{title}><span class="kpi-tag">{esc(tag)}</span>'
        f'<div class="kpi-val">{esc(value)}</div><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
    )


def rank_html(items) -> str:
    rows = []
    for it in items:
        title = esc(trunc(it["title"], 95))
        if it["link"]:
            t = f'<a href="{esc(it["link"])}" target="_blank" rel="noopener noreferrer" title="{esc(it["title"])}">{title}</a>'
        else:
            t = f'<span title="{esc(it["title"])}">{title}</span>'
        rows.append(
            f'<div class="rank-row" style="--w:{it["w"]:.1f}%"><div class="rank-n">{it["rank"]:02d}</div>'
            f'<div class="rank-t">{t}<div class="rank-m">{esc(it["meta"])}</div></div>'
            f'<div class="rank-v">{esc(it["value"])}</div></div>'
        )
    return squash('<div class="sawt rank">' + "".join(rows) + "</div>")


# =============================================================================
# 6. Plotly figures
# =============================================================================
def style_fig(fig: go.Figure, height: int = 380, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOT_FONT, color=C["ivory"], size=13),
        hoverlabel=dict(
            bgcolor=C["graphite"],
            bordercolor=C["bronze"],
            font=dict(family=PLOT_FONT, color=C["ivory"], size=13),
        ),
        showlegend=legend,
    )
    if legend:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=C["muted"], size=12))
        )
    axis = dict(
        gridcolor=GRID,
        zerolinecolor=AXIS_LINE,
        linecolor=AXIS_LINE,
        tickfont=dict(color=C["muted"], size=12),
        title_font=dict(color=C["muted"], size=12),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def fig_channel_bar(g: pd.DataFrame, col: str, xtitle: str, color: str, top_n: int = TOP_N_CHANNELS):
    g = g.sort_values(col, ascending=False).head(top_n)
    if g.empty:
        return None
    labels = [pesc(x) for x in g["channel_name"]]
    vals = g[col].to_numpy(dtype=float)
    custom = np.column_stack(
        [g["videos"].to_numpy(dtype=float), g["total_views"].to_numpy(dtype=float), g["median_views"].to_numpy(dtype=float)]
    )
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(color=color),
            text=[fmt_compact(v) for v in vals],
            textposition="outside",
            textfont=dict(color=C["ivory"], size=12),
            cliponaxis=False,
            customdata=custom,
            hovertemplate="<b>%{y}</b><br>Videos: %{customdata[0]:,.0f}<br>Total views: %{customdata[1]:,.0f}"
            "<br>Median views / video: %{customdata[2]:,.0f}<extra></extra>",
        )
    )
    style_fig(fig, height=max(240, 36 * len(g) + 70))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(title_text=xtitle, range=[0, float(vals.max()) * 1.22])
    return fig


def fig_channel_box(d: pd.DataFrame, min_videos: int, top_n: int = 10):
    counts = d["channel_name"].value_counts()
    eligible = counts[counts >= min_videos].head(top_n).index
    sub = d[d["channel_name"].isin(eligible) & (d["views"] > 0)]
    if sub.empty:
        return None
    order = sub.groupby("channel_name")["views"].median().sort_values(ascending=False).index
    fig = go.Figure()
    for ch in order:
        v = sub.loc[sub["channel_name"] == ch, "views"]
        fig.add_trace(
            go.Box(
                x=v,
                name=pesc(ch),
                orientation="h",
                boxpoints="outliers",
                marker=dict(color=C["amber"], size=4, opacity=0.7),
                line=dict(color=C["bronze"], width=1.4),
                fillcolor="rgba(231,169,75,0.16)",
                showlegend=False,
                hovertemplate="<b>" + pesc(ch) + "</b><br>Views: %{x:,.0f}<extra></extra>",
            )
        )
    style_fig(fig, height=max(300, 54 * len(order) + 90))
    fig.update_xaxes(type="log", title_text="Views (log scale)")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def fig_scatter(d: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, xlog: bool = True, ylog: bool = True):
    s = d[(d[xcol] > 0) & (d[ycol] > 0)]
    if len(s) < 5:
        return None
    custom = np.column_stack([s["title_hover"].to_numpy(), s["channel_hover"].to_numpy()])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s[xcol],
            y=s[ycol],
            mode="markers",
            name="مقطع",
            marker=dict(size=7, color=C["amber"], opacity=0.55, line=dict(width=0)),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            + xlabel
            + ": %{x:,.0f}<br>"
            + ylabel
            + ": %{y:,.0f}<extra></extra>",
        )
    )
    trend = binned_median(s[xcol], s[ycol])
    if trend is not None:
        fig.add_trace(
            go.Scatter(
                x=trend["x"],
                y=trend["y"],
                mode="lines+markers",
                name="وسيط الشرائح",
                line=dict(color=C["ivory"], width=2),
                marker=dict(size=6, color=C["ivory"]),
                hovertemplate="Binned median<br>" + xlabel + ": %{x:,.0f}<br>" + ylabel + ": %{y:,.0f}<extra></extra>",
            )
        )
    style_fig(fig, height=400, legend=trend is not None)
    fig.update_xaxes(type="log" if xlog else "linear", title_text=xlabel + (" (log scale)" if xlog else ""))
    fig.update_yaxes(type="log" if ylog else "linear", title_text=ylabel + (" (log scale)" if ylog else ""))
    return fig


def fig_heatmap(d: pd.DataFrame):
    cols = {"views": "Views", "likes": "Likes", "comments": "Comments", "duration_minutes": "Duration (min)"}
    sub = d[list(cols)].dropna()
    if len(sub) < MIN_CORR_SAMPLE:
        return None
    corr = sub.rank().corr()  # Pearson on ranks == Spearman
    labels = list(cols.values())
    z = corr.to_numpy(dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=CORR_SCALE,
            xgap=3,
            ygap=3,
            colorbar=dict(thickness=12, outlinewidth=0, tickfont=dict(color=C["muted"], size=11), title=dict(text="ρ")),
            hovertemplate="%{y} × %{x}<br>Spearman ρ = %{z:.2f}<extra></extra>",
        )
    )
    for i, yl in enumerate(labels):
        for j, xl in enumerate(labels):
            val = z[i][j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            dark = (not np.isnan(val)) and val > 0.62
            fig.add_annotation(
                x=xl, y=yl, text=txt, showarrow=False, font=dict(size=14, color=C["obsidian"] if dark else C["ivory"])
            )
    style_fig(fig, height=400)
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_duration_hist(d: pd.DataFrame):
    if len(d) < 5:
        return None
    med = float(d["duration_minutes"].median())
    fig = go.Figure(
        go.Histogram(
            x=d["duration_minutes"],
            nbinsx=20,
            marker=dict(color=C["amber"], line=dict(color=C["obsidian"], width=1)),
            opacity=0.92,
            hovertemplate="Duration: %{x} min<br>Videos: %{y}<extra></extra>",
        )
    )
    style_fig(fig, height=380)
    fig.add_vline(
        x=med,
        line_dash="dash",
        line_color=C["ivory"],
        line_width=1.2,
        annotation_text=f"Median {med:.0f} min",
        annotation_position="top right",
        annotation_font_color=C["ivory"],
        annotation_font_size=12,
    )
    fig.update_xaxes(title_text="Video duration (minutes)")
    fig.update_yaxes(title_text="Number of videos")
    return fig


def day_stats(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("publish_dow")["views"].agg(median="median", mean="mean", n="count").reindex(DAY_ORDER)
    g["n"] = g["n"].fillna(0).astype(int)
    return g


def fig_day(d: pd.DataFrame):
    g = day_stats(d)
    if g["n"].sum() < 5:
        return None
    labels = [DAY_AR[i] for i in DAY_ORDER]
    med = g["median"].to_numpy(dtype=float)
    colors = [C["amber"] if n >= MIN_DAY_SAMPLE else C["muted"] for n in g["n"]]
    custom = np.column_stack([g["n"].to_numpy(dtype=float), g["mean"].to_numpy(dtype=float)])
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=med,
            marker=dict(color=colors),
            text=[f"n={n}" for n in g["n"]],
            textposition="outside",
            textfont=dict(color=C["muted"], size=12),
            cliponaxis=False,
            customdata=custom,
            hovertemplate="%{x}<br>Median views: %{y:,.0f}<br>Mean views: %{customdata[1]:,.0f}"
            "<br>Videos: %{customdata[0]:,.0f}<extra></extra>",
        )
    )
    style_fig(fig, height=380)
    top = np.nanmax(med) if np.isfinite(med).any() else 1
    fig.update_yaxes(title_text="Median views", range=[0, float(top) * 1.2])
    fig.update_xaxes(title_text="Publication day")
    return fig


def fig_views_distribution(d: pd.DataFrame):
    v = d["views"][d["views"] > 0].to_numpy(dtype=float)
    if len(v) < 8:
        return None
    _, hi_f = iqr_fences(d["views"])
    lv = np.log10(v)
    if lv.max() - lv.min() < 1e-9:
        return None
    edges = np.linspace(lv.min(), lv.max(), 31)
    out_mask = v > hi_f
    cnt_all, _ = np.histogram(lv, bins=edges)
    cnt_out, _ = np.histogram(lv[out_mask], bins=edges)
    cnt_in = cnt_all - cnt_out
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges) * 0.92
    ranges = [f"{fmt_compact(10 ** a)} – {fmt_compact(10 ** b)}" for a, b in zip(edges[:-1], edges[1:])]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=cnt_in,
            width=widths,
            name="ضمن النطاق الإحصائي",
            marker=dict(color=C["bronze"]),
            customdata=ranges,
            hovertemplate="Views: %{customdata}<br>Videos: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=centers,
            y=cnt_out,
            width=widths,
            name="فوق حدّ IQR",
            marker=dict(color=C["amber"]),
            customdata=ranges,
            hovertemplate="Views: %{customdata}<br>Outlier videos: %{y}<extra></extra>",
        )
    )
    style_fig(fig, height=380, legend=True)
    fig.update_layout(barmode="stack", bargap=0.04)
    ticks = [t for t in range(int(math.floor(lv.min())), int(math.ceil(lv.max())) + 1) if lv.min() <= t <= lv.max()]
    fig.update_xaxes(tickvals=ticks, ticktext=[fmt_compact(10 ** t) for t in ticks], title_text="Views (log10 scale)")
    fig.update_yaxes(title_text="Number of videos")
    if hi_f > 0 and lv.min() < math.log10(hi_f) < lv.max():
        fig.add_vline(
            x=math.log10(hi_f),
            line_dash="dash",
            line_color=C["ivory"],
            line_width=1.2,
            annotation_text=f"Q3 + 1.5×IQR ≈ {fmt_compact(hi_f)}",
            annotation_position="top left",
            annotation_font_color=C["ivory"],
            annotation_font_size=12,
        )
    return fig


def fig_channel_compare(g: pd.DataFrame, min_videos: int):
    s = g[(g["videos"] >= min_videos) & (g["total_views"] > 0) & (g["median_views"] > 0)].copy()
    if len(s) < 2:
        return None
    sizes = (12 + 34 * np.sqrt(s["videos"] / s["videos"].max())).to_numpy(dtype=float)
    labelled = set(s.nlargest(8, "total_views").index)
    text = [pesc(n) if i in labelled else "" for i, n in zip(s.index, s["channel_name"])]
    custom = np.column_stack([[pesc(n) for n in s["channel_name"]], [str(int(v)) for v in s["videos"]]])
    fig = go.Figure(
        go.Scatter(
            x=s["median_views"],
            y=s["total_views"],
            mode="markers+text",
            text=text,
            textposition="top center",
            textfont=dict(color=C["ivory"], size=12),
            marker=dict(size=sizes, color=C["amber"], opacity=0.72, line=dict(color=C["bronze"], width=1)),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>Videos: %{customdata[1]}<br>Median views / video: %{x:,.0f}"
            "<br>Total views: %{y:,.0f}<extra></extra>",
        )
    )
    style_fig(fig, height=420)
    fig.update_xaxes(type="log", title_text="Median views per video (log scale)")
    fig.update_yaxes(type="log", title_text="Total views (log scale)")
    return fig


def render_figure(builder, key: str, *args, empty: str = "لا توجد بيانات كافية لهذا الرسم ضمن التحديد الحالي.", **kwargs) -> None:
    """Build + draw a chart. A failing chart shows a message instead of breaking the page."""
    try:
        fig = builder(*args, **kwargs)
        if fig is None:
            note(empty)
            return
        st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False, "responsive": True})
    except Exception as exc:  # noqa: BLE001 - keep the dashboard alive
        st.warning(f"تعذّر رسم هذا المخطط ({type(exc).__name__}: {exc})")


# =============================================================================
# 7. Statistics + insight texts
# =============================================================================
def channel_stats(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("channel_name", as_index=False).agg(
        videos=("video_id", "nunique"),
        total_views=("views", "sum"),
        median_views=("views", "median"),
    )


def insight_overview(d: pd.DataFrame) -> str:
    mean_v, med_v = float(d["views"].mean()), float(d["views"].median())
    body = (
        f"متوسط المشاهدات للمقطع <b>{fmt_int(mean_v)}</b> بينما وسيطها <b>{fmt_int(med_v)}</b>. "
        if med_v > 0
        else ""
    )
    if med_v > 0 and mean_v / med_v >= 2:
        body += (
            f"المتوسط يقارب <b>{mean_v / med_v:.1f}</b> أضعاف الوسيط، فالأعداد تتأثر بعدد قليل من المقاطع الواسعة الانتشار؛ "
            "لذلك نعتمد الوسيط والمقاييس اللوغاريتمية في قراءة المقطع «النموذجي»."
        )
    elif med_v > 0:
        body += "الفارق بينهما محدود ضمن التحديد الحالي، أي أن توزيع المشاهدات هنا أقل تأثراً بالقيم الضخمة."
    return body


def insight_channels(d: pd.DataFrame, min_videos: int) -> str:
    g = channel_stats(d)
    top_count = g.sort_values(["videos", "total_views"], ascending=False).iloc[0]
    top_total = g.sort_values("total_views", ascending=False).iloc[0]
    total = float(g["total_views"].sum())
    share = float(top_total["total_views"]) / total * 100 if total > 0 else 0.0
    if top_count["channel_name"] == top_total["channel_name"]:
        body = (
            f"<b>{esc(top_count['channel_name'])}</b> هي الأكثر حضوراً ضمن التحديد الحالي (<b>{int(top_count['videos'])}</b> مقطعاً) "
            f"والأعلى في إجمالي المشاهدات بحصة <b>{pct(share)}</b>. "
        )
    else:
        body = (
            f"أكثر القنوات حضوراً ضمن التحديد الحالي <b>{esc(top_count['channel_name'])}</b> بـ <b>{int(top_count['videos'])}</b> مقطعاً، "
            f"وتستحوذ <b>{esc(top_total['channel_name'])}</b> على <b>{pct(share)}</b> من إجمالي المشاهدات. "
        )
    eligible = g[g["videos"] >= min_videos]
    if len(eligible) >= 2:
        top_med = eligible.sort_values("median_views", ascending=False).iloc[0]
        body += (
            f"أما على مستوى المقطع النموذجي (القنوات ذات {min_videos} مقاطع فأكثر) فأعلى وسيط لدى "
            f"<b>{esc(top_med['channel_name'])}</b> بنحو <b>{fmt_int(top_med['median_views'])}</b> مشاهدة. "
        )
        if top_med["channel_name"] != top_total["channel_name"]:
            body += "الشعبية الإجمالية وأداء المقطع النموذجي ليسا الشيء نفسه."
    else:
        body += (
            f"لا توجد قنوات كافية بـ {min_videos} مقاطع فأكثر لحساب وسيط موثوق، "
            "فالأدلة غير كافية للمقارنة بين القنوات على مستوى المقطع."
        )
    return body


def _assoc(a: str, b: str, rho: float) -> str:
    if rho >= 0.2:
        return f"المقاطع الأعلى في {a} تميل غالباً إلى أن تكون الأعلى في {b}"
    if rho <= -0.2:
        return f"المقاطع الأعلى في {a} تميل غالباً إلى أن تكون الأقل في {b}"
    return f"لا يظهر ارتباط واضح بين {a} و{b}"


def insight_engagement(d: pd.DataFrame):
    rho_vl, n = spearman(d["views"], d["likes"])
    if rho_vl is None:
        return None
    rho_vc, _ = spearman(d["views"], d["comments"])
    rho_lc, _ = spearman(d["likes"], d["comments"])
    body = (
        f"ارتباط سبيرمان بين المشاهدات والإعجابات <b>ρ = {rho_vl:.2f}</b> ({strength_ar(rho_vl)}): "
        f"{_assoc('المشاهدات', 'الإعجابات', rho_vl)}. "
    )
    pair = d[["views", "likes"]].dropna()
    if len(pair) >= MIN_CORR_SAMPLE:
        pearson = float(pair["views"].corr(pair["likes"]))
        body += f"للمقارنة، Pearson على الأرقام الخام <b>{pearson:.2f}</b> وهو أكثر حساسية للمقاطع الضخمة. "
    if rho_vc is not None and rho_lc is not None:
        diff = rho_lc - rho_vc
        if abs(diff) < 0.03:
            body += f"ترتبط التعليقات بالمشاهدات (ρ = {rho_vc:.2f}) وبالإعجابات (ρ = {rho_lc:.2f}) بدرجة متقاربة. "
        elif diff > 0:
            body += f"ترتبط التعليقات بالإعجابات (ρ = {rho_lc:.2f}) أكثر من ارتباطها بالمشاهدات (ρ = {rho_vc:.2f}). "
        else:
            body += f"ترتبط التعليقات بالمشاهدات (ρ = {rho_vc:.2f}) أكثر من ارتباطها بالإعجابات (ρ = {rho_lc:.2f}). "
    body += "هذا ارتباط وصفي: لا يثبت أن أحد المقياسين يسبب الآخر."
    return body


def insight_duration(d: pd.DataFrame):
    rho, n = spearman(d["duration_minutes"], d["views"])
    if rho is None:
        return None
    lo, hi = d["duration_minutes"].min(), d["duration_minutes"].max()
    body = (
        f"وسيط مدة المقطع <b>{d['duration_minutes'].median():.0f}</b> دقيقة (النطاق من {lo:.0f} إلى {hi:.0f}). "
        f"ارتباط سبيرمان بين المدة والمشاهدات <b>ρ = {rho:.2f}</b> ({strength_ar(rho)}). "
    )
    if abs(rho) < 0.2:
        body += "لا يظهر ارتباط واضح بين طول المقطع وعدد مشاهداته ضمن هذا التحديد. "
    elif abs(rho) < 0.7:
        body += "المدة وحدها تفسّر جزءاً محدوداً من الفروق في المشاهدات. "
    body += "قد تعكس العلاقة اختلاف القنوات لا أثر الطول نفسه، ولا يمكن الاستنتاج أن إطالة الحلقة ترفع المشاهدات."
    return body


def insight_days(d: pd.DataFrame):
    g = day_stats(d)
    g = g[g["n"] >= MIN_DAY_SAMPLE]
    if len(g) < 2:
        return None
    best, worst = g["median"].idxmax(), g["median"].idxmin()
    return (
        f"بين أيام النشر التي تضم {MIN_DAY_SAMPLE} مقاطع فأكثر، كان أعلى وسيط مشاهدات ليوم <b>{DAY_AR[best]}</b> "
        f"(<b>{fmt_int(g.loc[best, 'median'])}</b> مشاهدة، n={int(g.loc[best, 'n'])}) وأدناه ليوم <b>{DAY_AR[worst]}</b> "
        f"(<b>{fmt_int(g.loc[worst, 'median'])}</b>، n={int(g.loc[worst, 'n'])}). "
        "هذا وصف للعينة فقط: لا يعني أن النشر في يوم بعينه يرفع المشاهدات، فالفروق قد تعكس اختلاف القنوات وأعمار المقاطع."
    )


def insight_outliers(d: pd.DataFrame):
    if len(d) < MIN_CORR_SAMPLE:
        return None
    _, hi = iqr_fences(d["views"])
    out = d[d["views"] > hi]
    total = float(d["views"].sum())
    top1 = float(d["views"].max()) / total * 100 if total > 0 else 0.0
    if out.empty:
        return (
            f"لا توجد مقاطع فوق حدّ القيم الشاذة الإحصائي (Q3 + 1.5×IQR ≈ <b>{fmt_int(hi)}</b> مشاهدة) ضمن التحديد الحالي؛ "
            f"وأعلى مقطع يمثل <b>{pct(top1, 1)}</b> من المشاهدات."
        )
    share = float(out["views"].sum()) / total * 100 if total > 0 else 0.0
    return (
        f"<b>{len(out)}</b> مقطعاً (<b>{pct(len(out) / len(d) * 100, 1)}</b>) تتجاوز حدّ القيم الشاذة الإحصائي "
        f"(Q3 + 1.5×IQR ≈ <b>{fmt_int(hi)}</b> مشاهدة) وتستحوذ على <b>{pct(share)}</b> من المشاهدات. "
        f"أعلى مقطع وحده يمثل <b>{pct(top1, 1)}</b>. هذه ملاحظات حقيقية لمقاطع واسعة الانتشار وليست أخطاء، "
        "ولذلك نفضّل الوسيط وارتباط سبيرمان والمقاييس اللوغاريتمية."
    )


# =============================================================================
# 8. Sidebar + filters
# =============================================================================
ALL_YEARS = "all"


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f["channels"]:
        mask &= df["channel_name"].isin(f["channels"])
    mask &= df["publish_dow"].isin(f["days"])
    mask &= df["duration_minutes"].between(f["duration"][0], f["duration"][1])
    if f["year"] != ALL_YEARS:
        mask &= df["publish_year"] == f["year"]
    return df[mask]


def reset_filters() -> None:
    """Reset every filter: new widget keys make Streamlit rebuild the widgets with their defaults."""
    st.session_state["reset_n"] = st.session_state.get("reset_n", 0) + 1


def widget_key(name: str, sig: str = "") -> str:
    return f"{name}_{sig}_{st.session_state.get('reset_n', 0)}"


def build_sidebar(df: pd.DataFrame, meta: dict):
    channels = df["channel_name"].value_counts().index.tolist()
    dmin, dmax = int(math.floor(df["duration_minutes"].min())), int(math.ceil(df["duration_minutes"].max()))
    years = sorted(df["publish_year"].unique().tolist(), reverse=True)
    # A different dataset gets different widget keys, so stale selections never leak in.
    sig = hashlib.md5("|".join(channels).encode("utf-8")).hexdigest()[:6] + f"{dmin}{dmax}{len(years)}"

    with st.sidebar:
        st.markdown(logo_html(compact=True), unsafe_allow_html=True)
        st.markdown(
            squash(
                '<div class="sawt side-h">عن البيانات</div>'
                f'<div class="sawt side-p">مقاطع بودكاست سعودية على يوتيوب (<b>{len(df):,}</b> مقطعاً من '
                f'<b>{df["channel_name"].nunique()}</b> قناة) جُمعت عبر YouTube Data API ثم نُظِّفت في الدفتر. '
                f"تشمل المقاطع التي مدتها {MIN_DURATION_MINUTES} دقيقة فأكثر؛ المشاهدات والإعجابات والتعليقات "
                "لقطة وقت الجمع"
                + (f" ({esc(meta['collected_on'])})" if meta.get("collected_on") else "")
                + ".</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sawt side-h">الفلاتر</div>', unsafe_allow_html=True)
        sel_channels = st.multiselect(
            "القنوات",
            options=channels,
            default=[],
            key=widget_key("f_channels", sig),
            placeholder="كل القنوات (اتركها فارغة)",
            help="اختر قناة أو أكثر، أو اتركها فارغة لعرض كل القنوات.",
        )
        sel_days = st.multiselect(
            "أيام النشر",
            options=DAY_ORDER,
            default=list(DAY_ORDER),
            format_func=lambda i: DAY_AR[i],
            key=widget_key("f_days", sig),
            help="يوم نشر المقطع بتوقيت الرياض.",
        )
        if dmin < dmax:
            sel_duration = st.slider(
                "مدة المقطع (بالدقائق)",
                min_value=dmin,
                max_value=dmax,
                value=(dmin, dmax),
                key=widget_key("f_duration", sig),
            )
        else:
            sel_duration = (dmin, dmax)
        sel_year = st.selectbox(
            "سنة النشر",
            options=[ALL_YEARS] + years,
            index=0,
            format_func=lambda v: "كل السنوات" if v == ALL_YEARS else str(v),
            key=widget_key("f_year", sig),
            help="المقاطع الأقدم أتيح لها وقت أطول لتجمع مشاهدات؛ اختر سنة واحدة لمقارنة مقاطع بعمر متقارب.",
        )
        st.button("إعادة ضبط الفلاتر", on_click=reset_filters)

    filters = {
        "channels": list(sel_channels),
        "days": list(sel_days),
        "duration": tuple(sel_duration),
        "year": sel_year,
    }
    d = apply_filters(df, filters)
    with st.sidebar:
        st.markdown(
            squash(
                f'<div class="sawt side-status">المعروض الآن: <b>{len(d):,}</b> من <b>{len(df):,}</b> مقطعاً '
                f'في <b>{d["channel_name"].nunique() if len(d) else 0}</b> قناة.</div>'
            ),
            unsafe_allow_html=True,
        )
    return filters, d


def filters_summary(f: dict, base: pd.DataFrame) -> str:
    chans = "كل القنوات" if not f["channels"] else "، ".join(f["channels"])
    days = "كل الأيام" if len(f["days"]) == len(DAY_ORDER) else ("، ".join(DAY_AR[i] for i in DAY_ORDER if i in f["days"]) or "لا شيء")
    year = "كل السنوات" if f["year"] == ALL_YEARS else str(f["year"])
    return f"القنوات: {chans} · الأيام: {days} · المدة: {f['duration'][0]}–{f['duration'][1]} دقيقة · السنة: {year}"


# =============================================================================
# 9. Page sections
# =============================================================================
def render_hero(base: pd.DataFrame, meta: dict) -> None:
    years = f"{int(base['publish_year'].min())} – {int(base['publish_year'].max())}"
    stats = [
        (f"{len(base):,}", "مقطعاً محلَّلاً"),
        (f"{base['channel_name'].nunique()}", "قناة ممثَّلة"),
        (years, "فترة النشر"),
    ]
    if meta.get("collected_on"):
        stats.append((meta["collected_on"], "تاريخ جمع البيانات"))
    stat_html = "".join(f'<div class="stat"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for v, k in stats)
    st.markdown(
        squash(
            f'<div class="sawt sawt-hero"><div class="hero-top">{logo_html()}'
            '<span class="hero-badge">Saudi Podcast Intelligence</span></div>'
            '<div class="hero-eyebrow">تحليل استكشافي · مقاطع البودكاست السعودي الطويلة على يوتيوب</div>'
            '<div class="hero-h1" role="heading" aria-level="1">خلف كل مشاهدة، <em>حكاية.</em></div>'
            '<p class="hero-p">لوحة تقرأ أداء مقاطع البودكاست السعودي كما تظهر في البيانات: أي القنوات تتصدّر، '
            "وكيف يتفاعل الجمهور مع الحلقات، وهل لطول الحلقة أو يوم نشرها علاقة بحجم الانتباه.</p>"
            '<p class="hero-q">السؤال المركزي: ما الذي يجعل بعض مقاطع البودكاست السعودي تجذب مشاهدات وتفاعلاً أكبر من غيرها؟</p>'
            '<div class="hero-cta"><a class="btn" href="#overview">استكشف البيانات<span aria-hidden="true">←</span></a></div>'
            f'<div class="hero-stats">{stat_html}</div>'
            f"{wave_html(data_wave(base))}"
            '<div class="wave-cap">موجة المشاهدات: كل عمود مجموعة من المقاطع المتتالية بتاريخ النشر، '
            "وارتفاعه وسيط مشاهداتها (مقياس لوغاريتمي).</div></div>"
        ),
        unsafe_allow_html=True,
    )
    links = "".join(f'<a href="#{a}">{esc(t)}</a>' for a, t in NAV)
    st.markdown(f'<nav class="sawt sawt-nav" aria-label="أقسام اللوحة">{links}</nav>', unsafe_allow_html=True)


def render_empty(filters: dict, base: pd.DataFrame) -> None:
    st.markdown(
        squash(
            '<div class="sawt empty"><b>لا توجد مقاطع تطابق الفلاتر الحالية</b>'
            "وسّع نطاق المدة، أو اختر سنة أخرى، أو اختر يوماً واحداً على الأقل، أو اضغط «إعادة ضبط الفلاتر» في الشريط الجانبي."
            f"<br>{esc(filters_summary(filters, base))}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_overview(d: pd.DataFrame, base: pd.DataFrame) -> None:
    section_header(
        "01",
        "نظرة عامة",
        "Executive overview",
        "صورة سريعة لما تبقّى من البيانات بعد الفلاتر. كل رقم يتحدّث مع الشريط الجانبي، ويحدّد وصفه إن كان إجمالياً أم وسيطاً.",
        "overview",
    )
    med_dur = float(d["duration_minutes"].median())
    cards = [
        kpi_card(0, "عدد", "المقاطع المحلَّلة", f"{len(d):,}", f"Videos · of {len(base):,} after notebook rules"),
        kpi_card(1, "إجمالي", "المشاهدات", fmt_compact(d["views"].sum()), "Total views", fmt_int(d["views"].sum())),
        kpi_card(2, "إجمالي", "الإعجابات", fmt_compact(d["likes"].sum()), "Total likes", fmt_int(d["likes"].sum())),
        kpi_card(3, "إجمالي", "التعليقات", fmt_compact(d["comments"].sum()), "Total comments", fmt_int(d["comments"].sum())),
        kpi_card(4, "وسيط", "مدة المقطع", f"{med_dur:.0f} min", f"Median duration · ≈ {med_dur / 60:.1f} h"),
        kpi_card(
            5,
            "وسيط",
            "المشاهدات لكل مقطع",
            fmt_compact(d["views"].median()),
            "Median views per video",
            fmt_int(d["views"].median()),
        ),
    ]
    st.markdown(squash('<div class="sawt kpis">' + "".join(cards) + "</div>"), unsafe_allow_html=True)
    insight_card("ما الذي يلفت النظر؟", insight_overview(d))


def render_channels(d: pd.DataFrame) -> int:
    section_header(
        "02",
        "مشهد القنوات",
        "Channel landscape",
        "من ينشر أكثر؟ ومن يجذب أكثر؟ نفرّق بين الشعبية الإجمالية (مجموع المشاهدات) وأداء المقطع النموذجي (الوسيط)؛ "
        "فالقناة الكثيرة المقاطع لا تتفوّق بالضرورة في كل مقطع.",
        "channels",
    )
    min_videos = st.slider(
        "الحد الأدنى لعدد مقاطع القناة (لحساب الوسيط والتوزيع)",
        min_value=1,
        max_value=10,
        value=DEFAULT_MIN_VIDEOS,
        key=widget_key("min_videos"),
        help=f"القيمة الافتراضية {DEFAULT_MIN_VIDEOS} كما في الدفتر: وسيط قناة بمقطع واحد لا يدل على أدائها المعتاد.",
    )
    g = channel_stats(d)
    insight_card("ماذا تقول البيانات؟", insight_channels(d, min_videos))

    left, right = st.columns(2, gap="large")
    with left:
        chart_head("عدد المقاطع لكل قناة", "Videos per channel")
        render_figure(fig_channel_bar, "ch_videos", g, "videos", "Number of videos", C["bronze"], 15)
    with right:
        chart_head("إجمالي المشاهدات لكل قناة", "Total views — popularity of the whole channel")
        render_figure(fig_channel_bar, "ch_total", g, "total_views", "Total views", C["amber"])

    left, right = st.columns(2, gap="large")
    with left:
        chart_head("وسيط المشاهدات لكل مقطع", f"Median views per video — channels with ≥ {min_videos} videos")
        eligible = g[g["videos"] >= min_videos]
        render_figure(
            fig_channel_bar,
            "ch_median",
            eligible,
            "median_views",
            "Median views per video",
            C["ivory"],
            empty=f"لا توجد قنوات بـ {min_videos} مقاطع فأكثر ضمن التحديد الحالي.",
        )
    with right:
        chart_head("توزيع المشاهدات عبر القنوات", f"Views per video — top 10 channels by video count, ≥ {min_videos} videos")
        render_figure(
            fig_channel_box,
            "ch_box",
            d,
            min_videos,
            empty=f"لا توجد قنوات بـ {min_videos} مقاطع فأكثر ضمن التحديد الحالي.",
        )
    return int(min_videos)


def render_engagement(d: pd.DataFrame) -> None:
    section_header(
        "03",
        "مختبر التفاعل",
        "Audience engagement lab",
        "هل تتحرك المشاهدات والإعجابات والتعليقات معاً؟ نستخدم مقياساً لوغاريتمياً لأن الأعداد شديدة التفاوت، "
        "ومعامل سبيرمان لأنه يعتمد على الترتيب فلا تخدعه المقاطع الضخمة.",
        "engagement",
    )
    text = insight_engagement(d)
    if text:
        insight_card("نظرة أقرب على التفاعل", text)
    else:
        insight_card("تنبيه", "عدد المقاطع غير كافٍ لحساب ارتباط موثوق ضمن التحديد الحالي؛ الأدلة غير كافية.", warn=True)

    left, right = st.columns(2, gap="large")
    with left:
        chart_head("المشاهدات مقابل الإعجابات", "Views vs Likes · log–log axes · line = median of quantile bins")
        render_figure(fig_scatter, "sc_vl", d, "views", "likes", "Views", "Likes")
    with right:
        chart_head("المشاهدات مقابل التعليقات", "Views vs Comments · log–log axes · videos with comments > 0")
        render_figure(fig_scatter, "sc_vc", d, "views", "comments", "Views", "Comments")

    left, right = st.columns(2, gap="large")
    with left:
        chart_head("الإعجابات مقابل التعليقات", "Likes vs Comments · log–log axes · videos with comments > 0")
        render_figure(fig_scatter, "sc_lc", d, "likes", "comments", "Likes", "Comments")
    with right:
        chart_head("مصفوفة ارتباط سبيرمان", "Spearman ρ · views, likes, comments, duration")
        render_figure(fig_heatmap, "heat", d)
    note(
        "ρ قريب من +1 يعني أن الترتيب يتفق تقريباً بين المقياسين، وقريب من 0 يعني غياب علاقة واضحة. "
        "الارتباط لا يعني السببية: كثرة الإعجابات قد ترافق كثرة المشاهدات دون أن تكون سببها."
    )


def render_timing(d: pd.DataFrame) -> None:
    section_header(
        "04",
        "المدة وأنماط النشر",
        "Duration & publishing patterns",
        "هل يرتبط طول الحلقة أو يوم نشرها بحجم المشاهدات؟ استخدم فلاتر المدة وأيام النشر والسنوات في الشريط الجانبي "
        "لاستكشاف نطاقات مختلفة. هذه علاقات وصفية داخل العينة ولا تثبت سببية.",
        "timing",
    )
    text = insight_duration(d)
    if text:
        insight_card("ماذا تقترح البيانات؟", text)

    left, right = st.columns(2, gap="large")
    with left:
        chart_head("توزيع مدة المقاطع", f"Video duration distribution · videos ≥ {MIN_DURATION_MINUTES} min only")
        render_figure(fig_duration_hist, "dur_hist", d)
    with right:
        chart_head("المدة مقابل المشاهدات", "Duration vs Views · log y-axis · line = median of quantile bins")
        render_figure(fig_scatter, "sc_dv", d, "duration_minutes", "views", "Duration (min)", "Views", xlog=False, ylog=True)

    chart_head("وسيط المشاهدات حسب يوم النشر", "Median views by publication day · gray bars = fewer than 5 videos")
    render_figure(fig_day, "day_bar", d)
    text = insight_days(d)
    if text:
        insight_card("قراءة بحذر", text, warn=True)
    else:
        note("لا توجد أيام نشر كافية (5 مقاطع فأكثر) للمقارنة بينها؛ الأدلة غير كافية.")


def render_signals(d: pd.DataFrame, min_videos: int) -> None:
    section_header(
        "05",
        "لحظات لافتة",
        "Featured signals",
        "المقاطع الأعلى مشاهدة والأعلى في معدل الإعجاب، والقيم الاستثنائية. الانتشار والتفاعل لا يقيسان جودة المحتوى.",
        "signals",
    )
    left, right = st.columns(2, gap="large")

    with left:
        chart_head("أعلى 10 مقاطع مشاهدة", "Top videos by views · bar length is proportional to views")
        top = d.nlargest(10, "views")
        vmax = float(top["views"].max()) or 1.0
        items = [
            {
                "rank": i + 1,
                "title": r["video_title"],
                "link": r["video_link"],
                "meta": f"{r['channel_name']} · {r['published_at']:%Y-%m-%d}",
                "value": fmt_compact(r["views"]),
                "w": float(r["views"]) / vmax * 100,
            }
            for i, (_, r) in enumerate(top.iterrows())
        ]
        st.markdown(rank_html(items), unsafe_allow_html=True)

    with right:
        eng = d[d["views"].gt(0) & d["likes"].notna()].copy()
        eng["like_rate"] = eng["likes"] / eng["views"] * 100
        min_views = float(eng["views"].median()) if len(eng) else 0.0
        chart_head("أعلى 10 مقاطع في معدل الإعجاب", f"Like rate = likes ÷ views · ranked among videos with ≥ {fmt_int(min_views)} views")
        elig = eng[eng["views"] >= min_views]
        top_rate = elig.nlargest(10, "like_rate")
        rmax = float(top_rate["like_rate"].max()) if len(top_rate) else 1.0
        items = [
            {
                "rank": i + 1,
                "title": r["video_title"],
                "link": r["video_link"],
                "meta": f"{r['channel_name']} · {fmt_compact(r['views'])} views",
                "value": f"{r['like_rate']:.2f}%",
                "w": float(r["like_rate"]) / (rmax or 1.0) * 100,
            }
            for i, (_, r) in enumerate(top_rate.iterrows())
        ]
        st.markdown(rank_html(items), unsafe_allow_html=True)
    note(
        "يُرتَّب معدل الإعجاب بين المقاطع التي مشاهداتها عند الوسيط أو أعلى (وسيط التحديد الحالي)، "
        "حتى لا تتصدّر القائمة مقاطع قليلة المشاهدات بمعدلات مضخّمة. الروابط تعمل فقط للمقاطع ذات المعرّف الصالح."
    )

    text = insight_outliers(d)
    left, right = st.columns(2, gap="large")
    with left:
        chart_head("القيم الاستثنائية في المشاهدات", "Views distribution · log10 scale · IQR rule from the notebook")
        render_figure(fig_views_distribution, "views_dist", d)
    with right:
        g = channel_stats(d)
        chart_head(
            "الإجمالي مقابل الوسيط لكل قناة",
            f"Total views vs median views per video · channels with ≥ {min_videos} videos · bubble size = number of videos",
        )
        render_figure(
            fig_channel_compare,
            "ch_compare",
            g,
            min_videos,
            empty=f"لا توجد قنوات كافية بـ {min_videos} مقاطع فأكثر ضمن التحديد الحالي.",
        )
    if text:
        insight_card("القيم الاستثنائية", text)
    note("المقاطع الفيروسية ملاحظات حقيقية في البيانات، لذلك لا تُحذف من التحليل ولا تُعامَل كأخطاء.")


def render_explorer(d: pd.DataFrame, base: pd.DataFrame, meta: dict, filters: dict) -> None:
    section_header(
        "06",
        "مستكشف البيانات",
        "Data explorer",
        "الجدول الخام والأعمدة والإحصاءات الوصفية، مع الفلاتر المطبّقة حالياً.",
        "explorer",
    )
    with st.expander("افتح الجدول والإحصاءات", expanded=True):
        c1, c2, c3 = st.columns(3)
        shown_cols = ["video_title", "channel_name", "published_at", "views", "likes", "comments", "duration_minutes", "video_link"]
        c1.markdown(f'<div class="sawt kpi"><div class="kpi-val">{len(d):,}</div><div class="kpi-label">الصفوف المعروضة</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="sawt kpi"><div class="kpi-val">{len(shown_cols)}</div><div class="kpi-label">الأعمدة المعروضة</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="sawt kpi"><div class="kpi-val">{len(base):,}</div><div class="kpi-label">صفوف مجموعة البيانات</div></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sawt note">الفلاتر الحالية: {esc(filters_summary(filters, base))}</div>',
            unsafe_allow_html=True,
        )

        preview = d[shown_cols].copy()
        preview["published_at"] = preview["published_at"].dt.strftime("%Y-%m-%d %H:%M")
        tab1, tab2, tab3 = st.tabs(["معاينة الجدول", "الأعمدة والأنواع", "إحصاءات وصفية"])
        with tab1:
            st.dataframe(
                preview,
                hide_index=True,
                column_config={
                    "video_title": st.column_config.TextColumn("العنوان"),
                    "channel_name": st.column_config.TextColumn("القناة"),
                    "published_at": st.column_config.TextColumn("تاريخ النشر"),
                    "views": st.column_config.NumberColumn("المشاهدات", format="%d"),
                    "likes": st.column_config.NumberColumn("الإعجابات", format="%d"),
                    "comments": st.column_config.NumberColumn("التعليقات", format="%d"),
                    "duration_minutes": st.column_config.NumberColumn("المدة (دقيقة)", format="%.1f"),
                    "video_link": st.column_config.LinkColumn("الرابط", display_text="يوتيوب"),
                },
            )
            st.download_button(
                "تنزيل البيانات المفلترة (CSV)",
                data=preview.to_csv(index=False).encode("utf-8-sig"),
                file_name="sawt_filtered_videos.csv",
                mime="text/csv",
            )
        with tab2:
            cols = [c for c in d.columns if c not in ("title_hover", "channel_hover")]
            info = pd.DataFrame(
                {
                    "العمود": cols,
                    "النوع": [str(d[c].dtype) for c in cols],
                    "غير الفارغ": [int(d[c].notna().sum()) for c in cols],
                    "المعنى": [COLUMN_DOCS.get(c, "") for c in cols],
                }
            )
            st.dataframe(info, hide_index=True)
        with tab3:
            desc = d[["views", "likes", "comments", "duration_minutes"]].describe().T.round(2)
            st.dataframe(desc)
        st.markdown(
            squash(
                '<div class="sawt note">قواعد الدفتر المطبَّقة: إزالة التكرار بمعرّف الفيديو · التعليقات المفقودة = 0 · '
                f"التوقيت Asia/Riyadh · الاحتفاظ بالمقاطع ≥ {MIN_DURATION_MINUTES} دقيقة "
                f"(الملف الأصلي {meta['raw_rows']:,} صفاً، أُزيل {meta['short_removed']:,} مقطعاً أقصر، "
                f"والمتبقي {meta['kept']:,}).</div>"
            ),
            unsafe_allow_html=True,
        )


def render_limits(base: pd.DataFrame, meta: dict) -> None:
    section_header(
        "07",
        "حدود التحليل",
        "Limitations",
        "ما الذي تقوله هذه البيانات، وما الذي لا تستطيع قوله.",
        "limits",
    )
    top_share = float(base["channel_name"].value_counts(normalize=True).iloc[0]) * 100
    top_channel = base["channel_name"].value_counts().index[0]
    total = float(base["views"].sum())
    top1 = float(base["views"].max()) / total * 100 if total > 0 else 0.0
    items = [
        f"العيّنة ناتجة عن بحث بكلمات مفتاحية في يوتيوب وليست حصراً شاملاً للبودكاست السعودي. "
        f"<b>{esc(top_channel)}</b> وحدها تمثّل <b>{pct(top_share)}</b> من المقاطع، فالنتائج العامة تعكس حضورها.",
        f"اقتُصر التحليل على المقاطع ({MIN_DURATION_MINUTES} دقيقة فأكثر) وفق قاعدة الدفتر؛ "
        f"استُبعد <b>{meta['short_removed']:,}</b> مقطعاً أقصر، فلا تنطبق النتائج على المقاطع القصيرة.",
        "المشاهدات تراكمية: المقاطع الأقدم أتيح لها وقت أطول، ولا نملك سلسلة زمنية يومية لكل مقطع؛ "
        "لذلك لا تُقرأ الفروق بين السنوات أو أيام النشر كأثر مباشر.",
        f"مقطع واحد يمثّل <b>{pct(top1, 1)}</b> من كل المشاهدات في البيانات. أُبقي في التحليل لأنه ملاحظة حقيقية، "
        "ولهذا تُستخدم الأوساط ومعامل سبيرمان والمقاييس اللوغاريتمية.",
        "الإعجابات والتعليقات والمشاهدات مقاييس انتباه وتفاعل، وليست مقاييس جودة. "
        "والقيم المفقودة في التعليقات عُوّضت بصفر كما في الدفتر.",
        "التحليل استكشافي وصفي: لا يوجد نموذج تنبؤي في هذا المشروع، والارتباطات المعروضة لا تثبت علاقات سببية.",
    ]
    cards = "".join(f'<div class="limit"><span class="ln">{i + 1:02d}</span>{t}</div>' for i, t in enumerate(items))
    st.markdown(squash(f'<div class="sawt limits">{cards}</div>'), unsafe_allow_html=True)
    st.markdown(
        squash(
            '<div class="sawt foot">SAWT | صوت — مشروع تحليل استكشافي لبيانات البودكاست السعودي على يوتيوب. '
            "مصدر البيانات: YouTube Data API v3. التصميم والتحليل لأغراض تعليمية.</div>"
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# 10. Entry point
# =============================================================================
def main() -> None:
    inject_css()
    try:
        df, meta = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    except FileNotFoundError:
        st.error(
            f"لم يُعثر على ملف البيانات: {DATA_FILE}\n\n"
            "ضع الملف clean_data.csv داخل المجلد data (أو بجانب app.py مباشرة)، أو حدّد مساراً آخر بالمتغير SAWT_DATA_FILE."
        )
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if df.empty:
        st.error("ملف البيانات لا يحتوي على مقاطع بعد تطبيق قواعد التنظيف.")
        st.stop()

    filters, d = build_sidebar(df, meta)
    render_hero(df, meta)

    if d.empty:
        render_empty(filters, df)
        st.stop()

    render_overview(d, df)
    min_videos = render_channels(d)
    render_engagement(d)
    render_timing(d)
    render_signals(d, min_videos)
    render_explorer(d, df, meta, filters)
    render_limits(df, meta)


main()
