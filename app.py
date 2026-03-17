import streamlit as st
import pandas as pd
import io
from utils.tools import answer_maintenance_question, analyse_sensor_data, diagnose_fault
from utils.sample_data import generate_sample_sensor_data

st.set_page_config(page_title="Predictive Maintenance Agent",
                   page_icon="⚙️", layout="wide")

st.title("⚙️ Predictive Maintenance AI Agent")
st.caption("Powered by GPT-3.5 · Domain: Rotating Machinery & Condition Monitoring")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Agent Tools")
    st.markdown("""
**Tool 1 — Domain Q&A**
Ask any question about maintenance, vibration, faults, or monitoring strategies.

**Tool 2 — Sensor Analysis**
Upload a CSV of sensor readings. The agent detects anomalies and assesses machine health.

**Tool 3 — Fault Diagnosis**
Describe symptoms. The agent diagnoses the fault, rates severity, and recommends action.
""")
    st.divider()
    st.subheader("Download sample data")
    col1, col2 = st.columns(2)
    for label, fault in [("Bearing fault", "bearing"),
                          ("Imbalance", "imbalance"),
                          ("Normal", "normal")]:
        csv = generate_sample_sensor_data(fault).to_csv(index=False)
        st.download_button(f"{label} CSV", csv,
                           f"sensor_{fault}.csv", "text/csv")
    st.divider()
    st.caption("Built with OpenAI + pandas + Streamlit")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Domain Q&A", "📊 Sensor Analysis", "🔍 Fault Diagnosis"])

# ── Tab 1: Domain Q&A ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ask the maintenance expert")
    suggestions = [
        "What are the ISO 10816 vibration severity limits for industrial machinery?",
        "How do I detect bearing defects using vibration analysis?",
        "What causes motor overheating and how can it be prevented?",
        "Explain the difference between predictive and preventive maintenance",
        "What is MTBF and how is it calculated?",
        "When should I replace a bearing vs lubricate it?",
    ]
    st.markdown("**Quick questions:**")
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"q{i}", use_container_width=True):
            st.session_state.qa_question = s

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    question = st.text_area("Your question:",
        value=st.session_state.get("qa_question", ""),
        height=80, placeholder="e.g. What vibration level indicates a critical bearing fault?")

    if st.button("Ask Agent", type="primary", key="ask_btn") and question.strip():
        st.session_state.qa_question = ""
        with st.spinner("Consulting maintenance knowledge base..."):
            result = answer_maintenance_question(question)
        st.session_state.qa_history.insert(0, {"q": question, "a": result["answer"]})

    for entry in st.session_state.qa_history:
        with st.expander(f"Q: {entry['q'][:80]}...", expanded=True if st.session_state.qa_history.index(entry)==0 else False):
            st.markdown(entry["a"])

# ── Tab 2: Sensor Analysis ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload sensor data for analysis")
    st.info("Upload a CSV with columns like: vibration_x, vibration_y, temperature, rpm, current_amp. "
            "Download sample files from the sidebar.")

    uploaded = st.file_uploader("Choose CSV file", type="csv", key="sensor_upload")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Raw data preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{len(df)} rows · {len(df.columns)} columns")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.subheader("Sensor trends")
            col_to_plot = st.selectbox("Select parameter to plot", numeric_cols)
            st.line_chart(df[col_to_plot])

        if st.button("Analyse with AI", type="primary", key="analyse_btn"):
            with st.spinner("Analysing sensor data..."):
                result = analyse_sensor_data(df)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Statistics")
                stats_df = pd.DataFrame(result["stats"]).T
                st.dataframe(stats_df, use_container_width=True)

            with col_b:
                st.subheader("Anomalies detected")
                if result["anomalies"]:
                    for a in result["anomalies"]:
                        color = "🔴" if a["severity"]=="HIGH" else "🟡"
                        st.markdown(f"{color} **{a['parameter']}** — "
                                    f"{a['anomaly_count']} outliers, "
                                    f"max deviation: {a['max_deviation']}")
                else:
                    st.success("No anomalies detected")

            st.subheader("AI Health Assessment")
            st.markdown(result["assessment"])

# ── Tab 3: Fault Diagnosis ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Describe symptoms for fault diagnosis")
    example_symptoms = [
        "High vibration at 1x RPM, slight noise from drive end bearing, temperature 15°C above baseline",
        "Intermittent high current draw, vibration increasing over last 2 weeks, oil contamination detected",
        "Loud knocking noise at startup, vibration spike at 2x RPM, bearing temperature 85°C",
        "Motor running hot, reduced output power, slight burning smell during operation",
    ]
    st.markdown("**Example symptom descriptions:**")
    for ex in example_symptoms:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state.symptoms = ex

    symptoms = st.text_area("Describe the symptoms:",
        value=st.session_state.get("symptoms", ""),
        height=100,
        placeholder="e.g. Increasing vibration on drive end, temperature 20°C above normal, "
                    "intermittent noise at high load...")

    if st.button("Diagnose Fault", type="primary", key="diagnose_btn") and symptoms.strip():
        st.session_state.symptoms = ""
        with st.spinner("Running fault diagnosis..."):
            result = diagnose_fault(symptoms)

        severity_colors = {
            "Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"
        }
        sev = result["severity"]
        icon = severity_colors.get(sev, "⚪")

        col1, col2, col3 = st.columns(3)
        col1.metric("Fault Type", result["fault_type"])
        col2.metric("Severity", f"{icon} {sev}")
        col3.metric("Immediate Action", "See below")

        st.divider()
        st.subheader("Full Diagnosis Report")
        st.markdown(result["raw"])

        if sev in ["Critical", "High"]:
            st.error(f"URGENT: {result['immediate_action']}")
        elif sev == "Medium":
            st.warning(f"ACTION REQUIRED: {result['immediate_action']}")
        else:
            st.info(f"RECOMMENDATION: {result['immediate_action']}")
