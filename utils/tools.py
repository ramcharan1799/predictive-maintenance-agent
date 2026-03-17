import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Tool 1: Domain Q&A ────────────────────────────────────────────────────────
def answer_maintenance_question(question: str) -> dict:
    """Answers predictive maintenance domain questions using GPT."""
    system = """You are a senior predictive maintenance engineer with 15 years of experience
in industrial machinery, rotating equipment, and condition monitoring.

You specialise in:
- Vibration analysis (FFT, RMS, peak values, bearing frequencies)
- Temperature monitoring and thermal anomalies
- Oil analysis and lubrication
- Motor current signature analysis
- Fault patterns: imbalance, misalignment, bearing defects, looseness
- Maintenance strategies: reactive, preventive, predictive, proactive
- Industry standards: ISO 10816, ISO 13373

Always structure your answer with:
1. Direct answer
2. Technical explanation
3. Key warning signs to watch
4. Recommended action

Be specific with numbers and thresholds where relevant."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=600
    )
    return {
        "tool": "domain_qa",
        "answer": response.choices[0].message.content.strip()
    }


# ── Tool 2: Sensor data analysis ─────────────────────────────────────────────
def analyse_sensor_data(df: pd.DataFrame) -> dict:
    """Analyses uploaded sensor CSV and detects anomalies."""
    stats = {}
    anomalies = []
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std  = df[col].std()
        mx   = df[col].max()
        mn   = df[col].min()
        stats[col] = {"mean": round(mean,2), "std": round(std,2),
                      "max": round(mx,2), "min": round(mn,2)}
        outliers = df[np.abs(df[col] - mean) > 3 * std]
        if len(outliers) > 0:
            anomalies.append({
                "parameter": col,
                "anomaly_count": len(outliers),
                "max_deviation": round(float(np.abs(outliers[col] - mean).max()), 2),
                "severity": "HIGH" if len(outliers) > 5 else "MEDIUM"
            })

    stats_str = "\n".join([
        f"{k}: mean={v['mean']}, std={v['std']}, min={v['min']}, max={v['max']}"
        for k,v in stats.items()
    ])
    anomaly_str = "\n".join([
        f"{a['parameter']}: {a['anomaly_count']} outliers, severity={a['severity']}"
        for a in anomalies
    ]) if anomalies else "No significant anomalies detected"

    prompt = f"""You are a condition monitoring engineer. Analyse these sensor statistics and anomalies.

Statistics:
{stats_str}

Anomalies detected:
{anomaly_str}

Provide:
1. Overall health assessment (Healthy / Warning / Critical)
2. Key findings from the data
3. Which parameters are concerning and why
4. Immediate recommended actions"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return {
        "tool": "sensor_analysis",
        "stats": stats,
        "anomalies": anomalies,
        "assessment": response.choices[0].message.content.strip()
    }


# ── Tool 3: Fault diagnosis ───────────────────────────────────────────────────
def diagnose_fault(symptoms: str) -> dict:
    """Diagnoses equipment fault from described symptoms."""
    system = """You are an expert fault diagnosis engineer for industrial rotating machinery.

Given symptom descriptions, provide a structured diagnosis.

Always respond in this exact format:
FAULT TYPE: [specific fault name]
SEVERITY: [Low / Medium / High / Critical]
CONFIDENCE: [percentage]
ROOT CAUSE: [2-3 sentence explanation]
WARNING SIGNS: [bullet list of 3-5 signs]
IMMEDIATE ACTION: [what to do right now]
MAINTENANCE PLAN: [scheduled actions with timeframes]
FAILURE RISK: [estimated time to failure if unaddressed]"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Equipment symptoms: {symptoms}"}
        ],
        temperature=0.1,
        max_tokens=500
    )
    raw = response.choices[0].message.content.strip()

    parsed = {}
    for line in raw.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            parsed[key.strip()] = val.strip()

    return {
        "tool": "fault_diagnosis",
        "raw": raw,
        "parsed": parsed,
        "severity": parsed.get("SEVERITY", "Unknown"),
        "fault_type": parsed.get("FAULT TYPE", "Unknown"),
        "immediate_action": parsed.get("IMMEDIATE ACTION", "Consult maintenance engineer")
    }
