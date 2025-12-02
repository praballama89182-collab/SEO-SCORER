# streamlit_seo_screener.py
"""
AKOI SEO CV SCREENER
Evaluates candidates for:
- On-page SEO
- Off-page / Link building
- Technical SEO
- Content & Blogging
- Analytics & Reporting
- SEO Tools

Features:
- 3D glass-style badge
- Gauge visualization
- Skills pie chart
- KPI bar chart
- 5-bullet auto summary
- CSV export
"""

import re
from io import BytesIO
from typing import List, Dict, Tuple
from datetime import datetime

import streamlit as st
import PyPDF2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------
# APP CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AKOI SEO CV SCREENER", layout="wide")
APP_TITLE = "AKOI SEO CV SCREENER"

# COLORS (NO RED ANYWHERE)
COLOR_LOW = "#FF9800"
COLOR_MED = "#FFD54F"
COLOR_HIGH = "#00C853"
COLOR_BLUE = "#1565C0"

PIE_PALETTE = [
    "#1565C0", "#00C853", "#FFA726", "#7B1FA2", "#0288D1", "#4CAF50"
]

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def hex_to_rgba(hex_color: str, alpha: float):
    r,g,b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------
# SEO KPI DEFINITIONS
# ---------------------------------------------------
KPIS = {
    "Technical SEO": [
        "crawl", "robots.txt", "sitemap", "canonical", "hreflang",
        "structured data", "schema", "json-ld",
        "core web vitals", "pagespeed", "lcp", "cls", "fid"
    ],
    "On-Page SEO": [
        "title tag", "meta description", "header tags", "h1", "h2",
        "keyword optimization", "internal linking", "image alt"
    ],
    "Off-Page / Link Building": [
        "backlink", "link building", "guest post", "outreach",
        "domain authority", "dr", "referring domains"
    ],
    "Content Strategy & Blogging": [
        "content strategy", "blog", "editorial", "content calendar",
        "pillar", "topic cluster", "long form", "content audit",
    ],
    "Analytics & Reporting": [
        "google analytics", "ga4", "search console", "gsc",
        "data studio", "reporting", "dashboard"
    ],
    "SEO Tools": [
        "semrush", "ahrefs", "screaming frog", "sitebulb",
        "moz", "majestic", "surfer seo", "seoquake"
    ],
}


TOOLS = {
    "Ahrefs": ["ahrefs"],
    "SEMrush": ["semrush"],
    "Screaming Frog": ["screaming frog"],
    "Search Console": ["search console", "gsc"],
    "Google Analytics": ["google analytics", "ga4"],
    "Content Tools": ["surfer seo", "marketmuse", "clearscope"]
}


# ---------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------
def extract_text_from_pdf(data: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(data))
    except:
        return ""
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t.lower() + "\n"
    return text


# ---------------------------------------------------
# EXPERIENCE EXTRACTION
# ---------------------------------------------------
def extract_years_experience(text: str) -> float:
    current = datetime.now().year
    explicit = re.findall(r"(\d{1,2}(?:\.\d)?)\s*(?:years|year|yrs)", text)
    since = re.findall(r"since\s+((?:19|20)\d{2})", text)
    exps = []

    for e in explicit:
        try: exps.append(float(e))
        except: pass

    for yr in since:
        try: exps.append(current - int(yr))
        except: pass

    return round(max(exps) if exps else 0, 1)


# ---------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------
def detect_numeric(text: str):
    return bool(re.search(r"\d+%|\d+k|\d+m|traffic|growth", text))

def score_keywords(text: str, keywords: List[str], numeric=True):
    found = sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", text))
    coverage = found / max(1, len(keywords))
    score = coverage * 8
    if numeric and detect_numeric(text):
        score += 2
    return round(min(score, 10), 1)


def score_all_kpis(text: str):
    return {k: score_keywords(text, v) for k,v in KPIS.items()}


def score_tools(text: str):
    out = {}
    for name, kw in TOOLS.items():
        out[name] = score_keywords(text, kw)
    return out


# ---------------------------------------------------
# SUMMARY GENERATION
# ---------------------------------------------------
def generate_summary(text, kpi_scores, tool_scores, years):
    bullets = []

    strong = [k for k,v in kpi_scores.items() if v >= 6]
    if strong:
        bullets.append("Strong in: " + ", ".join(strong[:3]))
    else:
        bullets.append("No standout SEO specialization detected.")

    if years >= 5:
        bullets.append(f"Experience: {years} yrs (Senior).")
    elif years >= 2:
        bullets.append(f"Experience: {years} yrs (Mid-level).")
    else:
        bullets.append(f"Experience: {years} yrs (Junior/Entry).")

    tools_used = [t for t,s in tool_scores.items() if s > 0]
    if tools_used:
        bullets.append("Tools: " + ", ".join(tools_used[:4]))
    else:
        bullets.append("Tools: No major SEO tools listed.")

    content_score = kpi_scores["Content Strategy & Blogging"]
    if content_score >= 6:
        bullets.append("Content Strategy: Strong blog/content background.")
    else:
        bullets.append("Content Strategy: Limited evidence.")

    risks = []
    if kpi_scores["Technical SEO"] < 4: risks.append("Weak technical SEO")
    if kpi_scores["Off-Page / Link Building"] < 4: risks.append("Weak link-building")
    if not tools_used: risks.append("Missing SEO tools")
    bullets.append("Risk: " + ", ".join(risks) if risks else "Risk: None major detected")

    bullets = bullets[:5]

    overview = (
        f"Candidate shows {years} yrs experience and strong areas in: "
        f"{', '.join(strong[:3]) or 'no major domains'}."
    )

    reason = "Validate portfolio, reporting samples, and hands-on case studies in interview."

    return bullets, overview, reason


# ---------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------
st.sidebar.markdown("### Weight Controls")

exp_w = st.sidebar.slider("Experience Weight", 10, 30, 20) / 100
core_w = st.sidebar.slider("Core SEO Weight", 25, 40, 30) / 100
content_w = st.sidebar.slider("Content Weight", 10, 30, 20) / 100
offpage_w = st.sidebar.slider("Off Page Weight", 5, 25, 15) / 100
tools_w = st.sidebar.slider("Tools Weight", 5, 15, 10) / 100

total = exp_w + core_w + content_w + offpage_w + tools_w
exp_w /= total; core_w /= total; content_w /= total; offpage_w /= total; tools_w /= total


# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------
st.title(APP_TITLE)
st.write("Evaluate SEO candidates automatically from their resume.")

file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
if not file:
    st.stop()

text = extract_text_from_pdf(file.read())
if not text.strip():
    st.error("Could not extract text. Upload a text-based PDF.")
    st.stop()


# ---------------------------------------------------
# COMPUTE SCORES
# ---------------------------------------------------
years = extract_years_experience(text)
exp_score = round(min((years ** 0.85) * 1.25, 10), 1)

kpi_scores = score_all_kpis(text)
core_avg = round(
    (kpi_scores["Technical SEO"] + kpi_scores["On-Page SEO"] + kpi_scores["Analytics & Reporting"]) / 3, 1
)

content_score = kpi_scores["Content Strategy & Blogging"]
offpage_score = kpi_scores["Off-Page / Link Building"]
tool_scores = score_tools(text)
tool_avg = round(sum(tool_scores.values()) / max(1, len(tool_scores)), 1)

combo = (
    exp_score * exp_w +
    core_avg * core_w +
    content_score * content_w +
    offpage_score * offpage_w +
    tool_avg * tools_w
)

best = max(exp_score, core_avg, content_score, offpage_score, tool_avg)
final_score = round(min(10, combo + 0.2 * (best - combo)), 2)


# ---------------------------------------------------
# DECISION LOGIC
# ---------------------------------------------------
if final_score < 3.7:
    decision = "Cannot Proceed"
    emoji = "🙁"
    badge_color = hex_to_rgba(COLOR_LOW, 0.9)
elif final_score < 5:
    decision = "Proceed to Next Round"
    emoji = "🙂"
    badge_color = hex_to_rgba(COLOR_MED, 0.9)
else:
    decision = "Good Candidate"
    emoji = "😄"
    badge_color = hex_to_rgba(COLOR_HIGH, 0.9)


# ---------------------------------------------------
# 3D GLASS BADGE
# ---------------------------------------------------
badge = f"""
<div style="
border-radius:12px; padding:14px; margin-bottom:10px;
background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01));
backdrop-filter: blur(6px); box-shadow: 0 8px 20px rgba(0,0,0,0.15);
border:1px solid rgba(255,255,255,0.06);
display:flex; justify-content:space-between; align-items:center;">
    <div style="display:flex; align-items:center;">
        <div style="width:60px;height:60px;border-radius:10px;background:{badge_color};
        display:flex;align-items:center;justify-content:center;font-size:28px;margin-right:12px;">
            {emoji}
        </div>
        <div>
            <div style="font-weight:700;font-size:17px;color:#0b2447">{decision}</div>
            <div style="font-size:13px;color:#294d61;">Final Score: {final_score} / 10</div>
        </div>
    </div>
</div>
"""
st.markdown(badge, unsafe_allow_html=True)


# ---------------------------------------------------
# GAUGE
# ---------------------------------------------------
g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=final_score,
    number={"suffix": " /10"},
    gauge={
        "axis": {"range": [0,10]},
        "bar": {"color": COLOR_HIGH if final_score>=5 else COLOR_MED if final_score>=3.7 else COLOR_LOW},
        "steps": [
            {"range": [0, 3.7], "color": hex_to_rgba(COLOR_LOW, 0.3)},
            {"range": [3.7, 5], "color": hex_to_rgba(COLOR_MED, 0.3)},
            {"range": [5, 10], "color": hex_to_rgba(COLOR_HIGH, 0.3)}
        ]
    }
))
g.update_layout(height=250)
st.plotly_chart(g, use_container_width=True)


# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------
bullets, overview, reason = generate_summary(text, kpi_scores, tool_scores, years)

st.subheader("Candidate Summary (5 Bullets)")
for b in bullets:
    st.markdown(f"- {b}")

st.markdown("### Overview")
st.write(overview)

st.markdown("### Notes")
st.write(reason)


# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------
c1, c2 = st.columns([2,1])

with c1:
    st.subheader("Tool/Skill Distribution")
    names = list(tool_scores.keys())
    vals = list(tool_scores.values())

    fig_pie = go.Figure(go.Pie(
        labels=names, values=vals, hole=0.3,
        marker=dict(colors=PIE_PALETTE)
    ))
    fig_pie.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("KPI Scores")
    df_kpi = pd.DataFrame(
        [[k, v] for k,v in kpi_scores.items()],
        columns=["KPI", "Score"]
    ).sort_values("Score", ascending=False)

    fig_bar = px.bar(
        df_kpi, x="Score", y="KPI", orientation="h",
        color="Score", color_continuous_scale=[COLOR_LOW,COLOR_MED,COLOR_HIGH],
        range_x=[0,10]
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("Components")
    comp = pd.DataFrame([
        ["Experience (yrs)", years],
        ["Experience Score", exp_score],
        ["Technical/On-Page/Analytics (avg)", core_avg],
        ["Content Score", content_score],
        ["Off-page Score", offpage_score],
        ["Tools Avg", tool_avg],
    ], columns=["Component","Value"])

    st.table(comp)

    st.markdown("### Final Decision")
    st.markdown(f"**{decision}** {emoji}")


# ---------------------------------------------------
# CSV EXPORT
# ---------------------------------------------------
final_df = df_kpi.copy()
final_df["Final Score"] = final_score
final_df["Decision"] = decision

st.download_button(
    "Download CSV Score Report",
    final_df.to_csv(index=False).encode("utf-8"),
    "seo_cv_report.csv",
    "text/csv"
)

st.caption("AKOI SEO CV Screener — Automated, heuristic, keyword-driven. Validate in interview." )
