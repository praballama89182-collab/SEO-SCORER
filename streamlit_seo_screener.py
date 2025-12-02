# streamlit_seo_screener.py
"""
AKOI SEO CV SCREENER
- Tailored for organic SEO profiles (on-page, off-page/linkbuilding, technical, content/blogging, analytics)
- Upload CV (PDF). App extracts text and scores candidate across SEO KPIs.
- Top 3D glass badge recommendation, gauge, skills pie, KPI bar chart.
- Replaces preview with 5-bullet summary.
- CSV download of KPI scores.
Dependencies: streamlit, PyPDF2, pandas, plotly
Run: streamlit run streamlit_seo_screener.py
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

# -------------------------
# App config & colors
# -------------------------
st.set_page_config(page_title="AKOI SEO CV SCREENER", layout="wide")
APP_TITLE = "AKOI SEO CV SCREENER"

# Friendly palette (avoid red)
COLOR_LOW = "#FF9800"    # orange
COLOR_MED = "#FFD54F"    # yellow
COLOR_HIGH = "#00C853"   # green
COLOR_ACCENT = "#1565C0" # blue
PIE_PALETTE = ["#1565C0", "#00C853", "#FFA726", "#7B1FA2", "#0288D1", "#4CAF50"]

def hex_to_rgb(hex_color: str) -> Tuple[int,int,int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    r,g,b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

# -------------------------
# KPI definitions (SEO)
# -------------------------
KPIS = {
    "Technical SEO & Site Health": [
        "crawl", "crawlability", "robots.txt", "sitemap", "sitemap.xml",
        "canonical", "hreflang", "structured data", "schema", "json-ld",
        "404", "redirect", "301", "302", "core web vitals", "lcp", "cls", "fid", "pagespeed"
    ],
    "On-Page SEO (Content & HTML)": [
        "title tag", "meta description", "header tags", "h1", "h2", "keyword optimization",
        "semantic html", "internal linking", "image alt", "internal link"
    ],
    "Off-Page & Link Building": [
        "backlink", "link building", "guest post", "outreach", "authority", "referral", "domain authority",
        "dr", "ahrefs", "link acquisition", "link profile", "link velocity"
    ],
    "Content Strategy & Blogging": [
        "blog", "editorial calendar", "content calendar", "content strategy", "pillar", "topic cluster",
        "content brief", "long-form", "evergreen", "surfer seo", "content gap", "content audit"
    ],
    "Analytics & Reporting": [
        "google analytics", "ga4", "search console", "gsc", "data studio", "lookerstudio",
        "kpi", "reporting", "a/b test", "experiment", "attribution"
    ],
    "Local SEO & Citations": [
        "google my business", "gmaps", "gmb", "local citations", "nap", "local seo", "maps"
    ],
    "SEO Tools & Automation": [
        "semrush", "ahrefs", "screaming frog", "deepcrawl", "sitebulb", "moz", "majestic", "seoquake",
        "python", "sql", "excel", "automation", "api"
    ],
    "E-E-A-T & Reputation": [
        "expert", "experience", "authoritative", "trust", "e-e-a-t", "eeat", "credentials", "author bio"
    ]
}

# Tools group for pie/skills distribution
TOOLS = {
    "Ahrefs": ["ahrefs", "dr "],
    "SEMrush": ["semrush"],
    "ScreamingFrog": ["screaming frog", "screamingfrog"],
    "Search Console": ["search console", "gsc"],
    "GA / Analytics": ["google analytics", "ga4"],
    "Content Tools": ["surfer seo", "clear scope", "marketmuse"]
}

# -------------------------
# Text extraction
# -------------------------
def extract_text_from_pdf(data: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(data))
    except Exception:
        return ""
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).lower()

# -------------------------
# Experience extractor (simple)
# -------------------------
def extract_years_experience(text: str) -> float:
    text = text.lower()
    current_year = datetime.now().year
    explicit = []
    for m in re.findall(r"(\d{1,2}(?:\.\d)?)\s*(?:years|year|yrs)", text):
        try:
            explicit.append(float(m))
        except:
            pass
    since = []
    for m in re.findall(r"since\s+((?:19|20)\d{2})", text):
        try:
            y = int(m); since.append(max(0, current_year-y))
        except:
            pass
    ranges = []
    for m in re.findall(r"((?:19|20)\d{2})\s*[-–to]{1,4}\s*((?:19|20)\d{2})", text):
        try:
            s,e = int(m[0]), int(m[1])
            if e>=s: ranges.append(e-s+1)
        except:
            pass
    candidates = []
    if explicit: candidates.append(max(explicit))
    if since: candidates.append(max(since))
    if ranges: candidates.append(sum(ranges))
    return round(max(candidates) if candidates else 0.0,1)

# -------------------------
# Scoring helpers
# -------------------------
def detect_numeric(text: str) -> bool:
    return bool(re.search(r"\d+\s?%|\d+k|\d+m|\bcreased by\b|\bboosted\b", text))

def score_keywords(text: str, keywords: List[str], numeric_boost: bool=True) -> Dict:
    found = []
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            found.append(kw)
    coverage = len(found)/max(1,len(keywords))
    base = coverage * 8.0
    bonus = 2.0 if (numeric_boost and detect_numeric(text)) else 0.0
    score_val = round(min(10.0, base+bonus),1)
    return {"score": score_val, "matches_count": len(found), "coverage": round(coverage,3)}

def score_all_kpis(text: str) -> Dict[str, Dict]:
    out = {}
    for k, kws in KPIS.items():
        out[k] = score_keywords(text, kws)
    return out

def score_tools(text: str) -> Dict[str,float]:
    out={}
    for name, kws in TOOLS.items():
        out[name] = score_keywords(text, kws)["score"]
    return out

# -------------------------
# Summary generator
# -------------------------
def generate_summary(text: str, kpi_scores: Dict[str,Dict], tools_scores: Dict[str,float], exp_years: float) -> Tuple[List[str], str, str]:
    bullets=[]
    # strengths: KPI categories >= 6
    strong = [k for k,v in kpi_scores.items() if v["score"]>=6.0]
    if strong:
        bullets.append("Strong in: " + ", ".join(strong[:3]) + ("" if len(strong)<=3 else " ..."))
    else:
        bullets.append("No major SEO domain dominance detected; may be generalist.")

    # experience
    if exp_years >= 5:
        bullets.append(f"Experience: {exp_years} years (senior-level).")
    elif exp_years >= 2:
        bullets.append(f"Experience: {exp_years} years (mid-level).")
    else:
        bullets.append(f"Experience: {exp_years} years (junior/entry).")

    # tools
    tool_list = [t for t,s in tools_scores.items() if s>0]
    if tool_list:
        bullets.append("Tools: " + ", ".join(tool_list[:4]))
    else:
        bullets.append("Tools: No major SEO tools mentioned.")

    # content/blogging signal
    content_score = kpi_scores.get("Content Strategy & Blogging", {}).get("score", 0)
    if content_score >= 6:
        bullets.append("Content & blogging: strong (experience in editorial/content strategy).")
    elif content_score >= 4:
        bullets.append("Content & blogging: some exposure; may need stronger portfolio.")
    else:
        bullets.append("Content & blogging: limited evidence.")

    # risk/gap
    risks=[]
    if kpi_scores.get("Technical SEO & Site Health",{}).get("score",0) < 4:
        risks.append("Technical SEO weak")
    if kpi_scores.get("Off-Page & Link Building",{}).get("score",0) < 4:
        risks.append("Link building experience limited")
    if not tool_list:
        risks.append("No tools listed")
    if risks:
        bullets.append("Risk: " + "; ".join(risks))
    else:
        bullets.append("No obvious red flags from CV text.")

    # trim/pad to 5 bullets
    bullets = bullets[:5]
    while len(bullets)<5:
        bullets.append("Additional details not detected in CV.")

    overview = f"Candidate shows {exp_years} yrs experience; top domains: {', '.join(strong[:3]) or 'none detected'}."
    reason = "Use this summary to prioritize interviews. Validate claims with evidence (reports, dashboards)."

    return bullets, overview, reason

# -------------------------
# Sidebar & weights
# -------------------------
st.sidebar.header("Settings")
numeric_boost = st.sidebar.checkbox("Enable numeric boost (+2 if numeric evidence present)", value=True)
exp_w = st.sidebar.slider("Experience weight %", 10,30,20)/100.0
core_w = st.sidebar.slider("Core SEO KPIs weight %", 25,45,30)/100.0
content_w = st.sidebar.slider("Content weight %", 10,30,20)/100.0
offpage_w = st.sidebar.slider("Off-page weight %", 5,25,15)/100.0
tools_w = st.sidebar.slider("Tools & Analytics weight %", 5,15,10)/100.0

# normalize to 1
total_w = exp_w + core_w + content_w + offpage_w + tools_w
exp_w /= total_w; core_w /= total_w; content_w /= total_w; offpage_w /= total_w; tools_w /= total_w

# -------------------------
# Main UI
# -------------------------
st.title(APP_TITLE)
st.markdown("Automated CV screener for organic SEO candidates (on-page, off-page, technical, content, analytics).")

uploaded = st.file_uploader("Upload CV (PDF - text-based)", type=["pdf"])
if not uploaded:
    st.info("Please upload a text-based PDF resume to begin.")
    st.stop()

try:
    file_bytes = uploaded.read()
except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

with st.spinner("Extracting text..."):
    text = extract_text_from_pdf(file_bytes)

if not text.strip():
    st.error("No text extracted (PDF may be scanned). Use OCR and re-upload.")
    st.stop()

# compute scores
years = extract_years_experience(text)
exp_score = round(min(10.0, (years ** 0.85) * 1.25),1)

kpi_scores = score_all_kpis(text)
# core marketplace (aggregate of technical + on-page + analytics)
core_avg = round((kpi_scores["Technical SEO & Site Health"]["score"] + kpi_scores["On-Page SEO (Content & HTML)"]["score"] + kpi_scores["Analytics & Reporting"]["score"]) / 3.0, 1)
content_score = kpi_scores["Content Strategy & Blogging"]["score"]
offpage_score = kpi_scores["Off-Page & Link Building"]["score"]
tools_scores = score_tools(text)
tools_avg = round(sum(tools_scores.values())/max(1,len(tools_scores)),1)

combined_raw = (
    exp_score * exp_w +
    core_avg * core_w +
    content_score * content_w +
    offpage_score * offpage_w +
    tools_avg * tools_w
)

# lenient nudge
best_comp = max(exp_score, core_avg, content_score, offpage_score, tools_avg)
final_score = round(min(10.0, combined_raw + 0.2*(best_comp - combined_raw)),2)

# thresholds: <3.7 Cannot proceed; 3.7–5 Proceed to next round; >=5 Good candidate
if final_score < 3.7:
    recommendation = "Cannot Proceed"
    emoji = "🙁"
    badge_color = hex_to_rgba(COLOR_LOW, 0.92)
elif 3.7 <= final_score < 5.0:
    recommendation = "Proceed to Next Round"
    emoji = "🙂"
    badge_color = hex_to_rgba(COLOR_MED, 0.92)
else:
    recommendation = "Good Candidate"
    emoji = "😄"
    badge_color = hex_to_rgba(COLOR_HIGH, 0.92)

# 3D glass-style badge (simple HTML)
card_html = f"""
<div style="
  border-radius:10px; padding:14px; margin-bottom:10px;
  background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01));
  backdrop-filter: blur(5px); box-shadow: 0 8px 20px rgba(15,23,42,0.14);
  border:1px solid rgba(255,255,255,0.04);
  display:flex; justify-content:space-between; align-items:center">
  <div style="display:flex; align-items:center;">
    <div style="width:60px; height:60px; border-radius:10px; background:{badge_color}; display:flex; align-items:center; justify-content:center; font-size:28px; margin-right:12px;">
      {emoji}
    </div>
    <div>
      <div style="font-weight:700; font-size:16px; color:#09223a;">{recommendation}</div>
      <div style="font-size:13px; color:#294b63;">Final score: <strong>{final_score} / 10</strong></div>
    </div>
  </div>
  <div style="text-align:right; color:#294b63;">
    <div style="font-size:12px">AKOI SEO CV SCREENER</div>
    <div style="font-size:11px; color:#4b6b8a">Automated screening — validate in interview</div>
  </div>
</div>
"""
st.markdown(card_html, unsafe_allow_html=True)

# Gauge
low_rgba = hex_to_rgba(COLOR_LOW, 0.3)
med_rgba = hex_to_rgba(COLOR_MED, 0.3)
high_rgba = hex_to_rgba(COLOR_HIGH, 0.3)
bar_color = COLOR_HIGH if final_score >= 5.0 else COLOR_MED if final_score >= 3.7 else COLOR_LOW

g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=final_score,
    number={'suffix': " /10", 'font': {'size':28}},
    gauge={
        'axis': {'range': [0,10]},
        'bar': {'color': bar_color},
        'steps': [
            {'range':[0,3.7], 'color': low_rgba},
            {'range':[3.7,5.0], 'color': med_rgba},
            {'range':[5.0,10], 'color': high_rgba}
        ],
        'threshold': {'line': {'color': "#000000", 'width': 3}, 'value': final_score}
    },
    title={'text': "Final SEO Score", 'font': {'size':13}}
))
g.update_layout(height=260, margin=dict(l=30,r=30,t=10,b=10))
st.plotly_chart(g, use_container_width=True)

# Summary (replace preview)
bullets, overview, reason = generate_summary(text, kpi_scores, tools_scores, years)
st.markdown("### Candidate Snapshot — 5 quick bullets")
for b in bullets:
    st.markdown(f"- {b}")

st.markdown("---")
st.markdown("**Overview:**")
st.write(overview)
st.markdown("**Notes / Reason:**")
st.write(reason)

# Charts and component table
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Skills / Tools Distribution")
    tool_names = list(tools_scores.keys())
    tool_vals = [tools_scores[n] for n in tool_names]
    if sum(tool_vals) == 0:
        tool_vals = [1]*len(tool_vals)
    color_seq = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(tool_names))]
    fig_pie = go.Figure(go.Pie(labels=tool_names, values=tool_vals, hole=0.3, marker=dict(colors=color_seq, line=dict(color="#ffffff", width=1))))
    fig_pie.update_traces(textinfo='label+percent', textposition='inside')
    fig_pie.update_layout(height=420)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("KPI Scores (Detected)")
    rows=[]
    for k,v in kpi_scores.items():
        rows.append([k, v["score"], v["coverage"]])
    kpi_df = pd.DataFrame(rows, columns=["KPI","Score","Coverage"]).sort_values("Score", ascending=False)
    fig_bar = px.bar(kpi_df, x="Score", y="KPI", orientation='h', color="Score", range_x=[0,10], color_continuous_scale=[COLOR_LOW, COLOR_MED, COLOR_HIGH])
    fig_bar.update_layout(height=520, margin=dict(l=260,t=30))
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Component Scores & Details")
    comp_df = pd.DataFrame([
        ["Experience (yrs)", years],
        ["Experience Score", exp_score],
        ["Technical & Analytics (avg)", core_avg],
        ["Content (blogging)", content_score],
        ["Off-page / Link Building", offpage_score],
        ["Tools & Automation (avg)", tools_avg],
    ], columns=["Component","Value"])
    st.table(comp_df.style.format({"Value":"{:.1f}"}))

    st.markdown("---")
    st.subheader("Recommendation")
    st.markdown(f"**{recommendation}** {emoji}")
    st.markdown(f"Final score: **{final_score} / 10**")

# CSV export of KPI scores
export_df = kpi_df.copy()
export_df = export_df.rename(columns={"Score":"KPI_Score","Coverage":"KPI_Coverage"})
export_df["Final_Score"] = final_score
export_df["Recommendation"] = recommendation
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download scores CSV", csv_bytes, file_name="seo_cv_scores.csv", mime="text/csv")

st.caption("Automated heuristic screening — verify claims, request dashboards and sample work during interview.")
