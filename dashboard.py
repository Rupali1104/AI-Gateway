"""Streamlit log viewer — shows request log table with key metadata."""

import pandas as pd
import streamlit as st

import logger

st.set_page_config(page_title="AI Gateway — Log Viewer", layout="wide")
st.title("AI Gateway — Request Log")

if st.button("Refresh"):
    st.rerun()

records = logger.read_all()

if not records:
    st.info("No requests logged yet. Send some prompts to POST /chat first.")
    st.stop()

df = pd.DataFrame(records)

cols = ["timestamp", "prompt_snippet", "model", "routing_reason",
        "routing_score", "routing_confidence", "latency_ms", "cache_hit",
        "similarity", "tokens"]
df = df.reindex(columns=[c for c in cols if c in df.columns])
df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

# Fill NaN so they display as blank strings
df = df.fillna("")

# ── Summary metrics ───────────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requests", len(df))
cache_hits = (df["cache_hit"] == True).sum() if "cache_hit" in df else 0
col2.metric("Cache hits", int(cache_hits))
col3.metric("Cache hit rate", f"{100*cache_hits/len(df):.1f}%" if len(df) else "—")
if "model" in df:
    fast = df["model"].str.contains("Fast", na=False).sum()
    col4.metric("Fast / Capable", f"{fast} / {len(df) - fast}")
st.divider()

# ── HTML table with explicit row colouring ────────────────────────────────────
header_cols = [c for c in cols if c in df.columns]

header_html = "".join(
    f"<th style='background:#2c3e50;color:white;padding:8px 12px;text-align:left;white-space:nowrap'>{c}</th>"
    for c in header_cols
)

rows_html = ""
for _, row in df.iterrows():
    is_hit = row.get("cache_hit") == True
    bg = "#c8f7c5" if is_hit else "#ffffff"
    cells = ""
    for c in header_cols:
        val = row[c]
        # Truncate long routing_reason for readability
        if c == "routing_reason" and isinstance(val, str) and len(val) > 60:
            val = val[:60] + "..."
        cells += f"<td style='padding:7px 12px;color:#111;border-bottom:1px solid #ddd;white-space:nowrap'>{val}</td>"
    rows_html += f"<tr style='background:{bg}'>{cells}</tr>"

table_html = f"""
<div style='overflow-x:auto'>
<table style='border-collapse:collapse;width:100%;font-size:13px;font-family:monospace'>
  <thead><tr>{header_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
<p style='font-size:12px;color:gray;margin-top:6px'>
  Green rows = cache hit &nbsp;|&nbsp; White rows = live LLM call
</p>
"""

st.markdown(table_html, unsafe_allow_html=True)
