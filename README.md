# Predictive Maintenance AI Agent

An AI-powered agent for industrial equipment health monitoring, fault diagnosis, and maintenance decision support. Built using GPT-3.5 with domain-specific prompt engineering for rotating machinery and condition monitoring.

## Tools

**Tool 1 — Domain Q&A**
Ask any question about predictive maintenance, vibration analysis, bearing faults, lubrication, ISO standards, and maintenance strategies. Answers follow a structured format: direct answer, technical explanation, warning signs, recommended action.

**Tool 2 — Sensor Data Analysis**
Upload a CSV of sensor readings (vibration, temperature, RPM, current). The agent computes statistics, detects anomalies using 3-sigma outlier detection, and generates an AI health assessment (Healthy / Warning / Critical).

**Tool 3 — Fault Diagnosis**
Describe equipment symptoms in plain English. The agent returns a structured diagnosis: fault type, severity, confidence, root cause, warning signs, immediate action, maintenance plan, and failure risk timeline.

## Sample data included

Download from the sidebar:
- `sensor_bearing.csv` — bearing fault with injected vibration and temperature anomalies
- `sensor_imbalance.csv` — rotor imbalance pattern
- `sensor_normal.csv` — healthy machine baseline

## Tech stack

- **OpenAI GPT-3.5** — domain Q&A, sensor assessment, fault diagnosis
- **pandas + numpy** — sensor data processing, 3-sigma anomaly detection
- **Streamlit** — tabbed UI, file upload, metrics, line charts
- **python-dotenv** — API key management

## Project structure
```
predictive-maintenance-agent/
├── app.py                  # Streamlit UI — 3 tabs, one per tool
├── utils/
│   ├── tools.py            # 3 agent tools: QA, sensor analysis, fault diagnosis
│   └── sample_data.py      # Realistic sensor data generator with injected faults
└── requirements.txt
```

## Run locally
```
git clone https://github.com/ramcharan1799/predictive-maintenance-agent
cd predictive-maintenance-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
streamlit run app.py
```

## Domain coverage

- Vibration analysis (RMS, peak, 1x/2x RPM patterns)
- Bearing fault detection and diagnosis
- ISO 10816 vibration severity standards
- Temperature monitoring and thermal anomalies
- Motor current signature analysis
- Maintenance strategies: reactive, preventive, predictive, proactive
- MTBF, MTTR, OEE concepts

## What I learned

- How to build a multi-tool AI agent with domain-specific system prompts
- How to use statistical methods (3-sigma) for anomaly detection in sensor data
- How to engineer structured output prompts for consistent, parseable LLM responses
- How to combine domain expertise (ECE/EE background) with LLM capabilities

## Author

Built as part of my AI Engineer learning journey — roadmap.sh/ai-engineer
MS Electrical Engineering | B.Tech ECE | Data Engineering background
