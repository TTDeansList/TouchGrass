# frontend_streamlit/app.py
import os
import time
import json
import requests
import pandas as pd
import streamlit as st

# ---------- Config ----------
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
APP_TITLE = "ReviewGuard by Team Touch Grass"
APP_DESC = "Classify reviews for quality, relevency and policy violations here."

st.set_page_config(
    page_title="ReviewGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Minimal CSS polish ----------
st.markdown(
    """
    <style>
      /* Global */
      .main { padding-top: 1rem; }
      section[data-testid="stSidebar"] { width: 340px !important; }
      /* Headings */
      h1, h2, h3 { letter-spacing: 0.2px; }
      /* Cards */
      .card {
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.06);
        padding: 18px 18px 14px 18px;
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(6px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }
      .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(0,0,0,0.08);
        margin-right: 6px;
      }
      .badge-ok { background: #ECFDF5; color: #065F46; }
      .badge-flag { background: #FEF2F2; color: #991B1B; }
      .muted { color: #6b7280; }
      .mono { font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; }
      .footnote { color:#6b7280; font-size:12px; }
      .pill {
        padding: 6px 12px; border-radius: 999px; font-size: 12px;
        border: 1px dashed rgba(0,0,0,0.1); background: #F9FAFB;
      }
      .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; }
      .grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    backend = st.text_input("Backend URL", value=DEFAULT_BACKEND, help="Your FastAPI server base URL.")
    colA, colB = st.columns([1,1])
    with colA:
        ping = st.button("🔌 Check Health")
    with colB:
        clear_hist = st.button("🧹 Clear History")
    st.divider()
    st.markdown("#### 💡 Tips")
    st.markdown(
        """
        <div class='pill mono'>Optional zero-shot</div>
        <span class='footnote'>Start backend with:<br>
        <code>export HF_ZEROSHOT=1</code><br>
        <code>export HF_ZS_MODEL=facebook/bart-large-mnli</code></span>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Health endpoint: `GET /health`")

# ---------- Utilities ----------
def call_health(url: str):
    try:
        r = requests.get(f"{url}/health", timeout=8)
        if r.status_code == 200:
            return True, r.json()
        return False, {"error": f"HTTP {r.status_code}", "text": r.text[:200]}
    except Exception as e:
        return False, {"error": str(e)}

def call_predict(url: str, text: str):
    r = requests.post(f"{url}/predict", json={"text": text}, timeout=15)
    r.raise_for_status()
    return r.json()

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

if clear_hist:
    st.session_state.history = []

# ---------- Header ----------
st.markdown(f"## 🛡️ {APP_TITLE}")
st.markdown(f"<span class='muted'>{APP_DESC}</span>", unsafe_allow_html=True)

# Health ping
if ping:
    ok, info = call_health(backend)
    if ok:
        cols = st.columns(3)
        with cols[0]:
            st.success("Backend: Online")
        with cols[1]:
            st.info(f"Model loaded: **{info.get('model_loaded')}**")
        with cols[2]:
            zs = info.get("hf_zeroshoot_enabled")
            st.warning(f"Zero-shot: **{zs}**" if zs else "Zero-shot: **False**")
        with st.expander("Health JSON"):
            st.json(info)
    else:
        st.error("Backend not reachable")
        with st.expander("Details"):
            st.json(info)

st.divider()

# ---------- Tabs ----------
tab_single, tab_batch, tab_about = st.tabs(["🔎 Single Review", "🗂️ Batch (CSV)", "ℹ️ About"])

# --- Single Review Tab ---
with tab_single:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Enter Review Text")
        text = st.text_area(
            " ",
            height=160,
            placeholder="e.g. I haven't been here but it looks terrible...",
            label_visibility="collapsed",
        )
        examples = st.multiselect(
            "Quick Examples",
            [
                "Buy my course at www.spam.com, best discount now!",
                "Great ambience and kind staff; would return.",
                "I haven't been here but I heard it's terrible.",
                "Use code SAVE20 for 10% off — visit http://promo.deals",
                "This is unrelated to the cafe.",
            ],
            default=[],
            help="Add one or more quick examples to the text box.",
        )
        if examples:
            text = (text + " " + " ".join(examples)).strip()

        go = st.button("🚀 Run Classification", type="primary")
        if go:
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Scoring…"):
                    try:
                        data = call_predict(backend, text)
                        # Append to history
                        st.session_state.history.insert(0, {
                            "ts": time.strftime("%H:%M:%S"),
                            "input": text,
                            "output": data
                        })
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    with right:
        st.markdown("#### Result")
        if st.session_state.history:
            latest = st.session_state.history[0]["output"]
            action = latest.get("action", "ok")
            proba = latest.get("quality_violation_prob", 0.0)
            flags = latest.get("flags", {}) or {}
            zs = latest.get("zeroshot", None)

            # Summary card
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            badge = "badge-flag" if action == "flag" else "badge-ok"
            st.markdown(
                f"<span class='badge {badge}'>Action: {action.upper()}</span> "
                f"<span class='badge'>Prob: {proba:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(proba, 0.0), 1.0))
            st.markdown("</div>", unsafe_allow_html=True)

            # Flags
            st.markdown("##### Policy Flags")
            if flags:
                cols = st.columns(len(flags))
                for i,(k,v) in enumerate(flags.items()):
                    with cols[i]:
                        st.metric(k.replace("_"," ").title(), "True" if v else "False")
            else:
                st.caption("No flags returned.")

            # Zero-shot scores
            if zs is not None:
                st.markdown("##### Zero-shot Category Scores")
                zs_pairs = sorted(zs.items(), key=lambda kv: kv[1], reverse=True)
                for label,score in zs_pairs:
                    st.write(f"• **{label}** — {score:.3f}")

            with st.expander("Raw JSON"):
                st.json(latest)
        else:
            st.caption("Run a classification to see results here.")

    st.markdown("#### History")
    if st.session_state.history:
        for item in st.session_state.history[:6]:
            out = item["output"]
            action = out.get("action","ok")
            badge = "badge-flag" if action == "flag" else "badge-ok"
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='badge {badge}'>Action: {action.upper()}</span> "
                f"<span class='badge'>Prob: {out.get('quality_violation_prob',0.0):.3f}</span> "
                f"<span class='muted'>@ {item['ts']}</span>",
                unsafe_allow_html=True,
            )
            st.write(item["input"])
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("No history yet.")

# --- Batch Tab ---
with tab_batch:
    st.markdown("#### Upload CSV")
    st.caption("CSV must have a `text` column. We’ll call the backend for each row and show a scored table.")
    file = st.file_uploader("Drag & drop a CSV", type=["csv"])
    run_batch = st.button("📥 Process CSV")
    if run_batch and file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            results = []
            bar = st.progress(0)
            for i, t in enumerate(df["text"].astype(str).tolist(), start=1):
                try:
                    data = call_predict(backend, t)
                    results.append({
                        "text": t,
                        "action": data.get("action"),
                        "prob": data.get("quality_violation_prob"),
                        "ad_spam": data.get("flags", {}).get("ad_spam"),
                        "irrelevant": data.get("flags", {}).get("irrelevant"),
                        "non_visit_rant": data.get("flags", {}).get("non_visit_rant"),
                    })
                except Exception as e:
                    results.append({"text": t, "action": "error", "prob": None, "ad_spam": None, "irrelevant": None, "non_visit_rant": None})
                bar.progress(min(i / len(df), 1.0))
            st.success(f"Processed {len(results)} rows.")
            out_df = pd.DataFrame(results)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "⬇️ Download results as CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="reviews_scored.csv",
                mime="text/csv",
            )

# --- About Tab ---
with tab_about:
    st.markdown("#### What’s happening under the hood")
    st.write("""
- **Baseline ML:** TF-IDF + Logistic Regression predicts overall violation probability.
- **Policy Rules:** Lightweight regex flags for ads/spam, irrelevant content, and non-visit rants.
- **Zero-shot (optional):** If enabled on the backend, we also return category scores via a HF model (e.g., `facebook/bart-large-mnli`).
- **Action:** We suggest `FLAG` if (probability ≥ threshold) OR any policy rule is triggered.
""")
    st.caption("Start backend with zero-shot: `export HF_ZEROSHOT=1; export HF_ZS_MODEL=facebook/bart-large-mnli`")