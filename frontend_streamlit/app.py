import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ReviewGuard Demo", page_icon="🔍", layout="centered")
st.title("🔍 Filtering the Noise — ReviewGuard Demo")
st.caption("Paste a review, get policy flags and a suggested action. (Zero-shot optional).")

with st.sidebar:
    st.subheader("Settings")
    backend = st.text_input("Backend URL", BACKEND_URL)
    st.caption("Backend dev server: uvicorn backend.src.main:app --reload")
    st.markdown("---")
    st.markdown("**Tips**")
    st.write(
        "• Optional zero-shot: export HF_ZEROSHOT=1; export HF_ZS_MODEL=facebook/bart-large-mnli\n"
        "• Health: GET /health"
    )

st.markdown("### Enter a review")
text = st.text_area("Review text", height=140, placeholder="e.g. I haven't been here but it looks terrible...")

col1, col2 = st.columns([1,1])
with col1:
    run = st.button("Run Classification", type="primary")
with col2:
    example = st.button("Load Example")

if example:
    text = "Buy my course at www.spam.com, best discount now!"

if run:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            resp = requests.post(f"{backend}/predict", json={"text": text}, timeout=15)
            if resp.status_code != 200:
                st.error(f"Backend error [{resp.status_code}]: {resp.text[:500]}")
            else:
                data = resp.json()
                st.markdown("### Raw JSON")
                st.json(data)

                # Pretty summary
                st.markdown("#### Summary")
                action = data.get("action", "ok")
                proba = data.get("quality_violation_prob", 0.0)
                flags = data.get("flags", {}) or {}
                zs = data.get("zeroshot", None)

                if action == "flag":
                    st.error(f"**Action:** {action.upper()}  |  **Model Prob:** {proba:.3f}")
                else:
                    st.success(f"**Action:** {action.upper()}  |  **Model Prob:** {proba:.3f}")

                st.markdown("##### Policy Flags")
                if not flags:
                    st.write("No flags provided.")
                else:
                    cols = st.columns(len(flags))
                    for i, (k, v) in enumerate(flags.items()):
                        with cols[i]:
                            st.metric(k.replace("_", " ").title(), "True" if v else "False")

                if zs is not None:
                    st.markdown("##### Zero-shot Category Scores")
                    for label, score in sorted(zs.items(), key=lambda kv: kv[1], reverse=True):
                        st.write(f"- **{label}**: {score:.3f}")

                with st.expander("What the rules mean"):
                    st.markdown("""
- **Advertisement/Spam**: URLs or promo language (*promo, discount, use code*).
- **Irrelevant**: Off-topic content not about the location.
- **Rant Without Visit**: Self-declared non-visit complaints (*never been here*).
                    """)
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("Built for TikTok TechJam by Team Touch Grass with baseline TF-IDF + Logistic Regression + rule-based policy checks, optional zero-shot enrichment.")