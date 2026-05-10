# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

智能装备健康管理系统（IEHM） — 工业设备预测性维护与智能诊断平台。模拟 3 台设备（数控机床/织布机/机械臂）的传感器数据，训练 XGBoost 模型预测剩余使用寿命（RUL），通过大模型 API 生成故障诊断报告，并以工业级 HMI 风格的数字孪生看板展示。

## Tech Stack

- **Language:** Python 3.10+
- **Data:** NumPy, Pandas, Matplotlib, Seaborn
- **ML:** XGBoost, scikit-learn
- **LLM:** OpenAI-compatible API (via `requests` or `openai` library)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** HTML5 + CSS3 + JavaScript + ECharts (纯静态，无框架)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sensor data and train model
python main.py --generate-data --train

# Start dashboard server
python main.py --dashboard
# Access http://localhost:8000

# Full pipeline
python main.py --all

# Run tests
pytest tests/ -v
```

## Architecture

5 modules in a single Python project:

1. **`data/`** — Sensor data simulation and fault injection. Generates 7-day time-series for 3 devices with 4 sensor types (temp/vibration/current/speed). Fault modes: bearing wear, overheating, rotor imbalance, belt loosening.

2. **`models/`** — XGBoost regression for RUL prediction. Feature engineering uses rolling window statistics (60-step). 5-fold time-series cross-validation. Outputs R²/RMSE/MAE metrics and feature importance rankings.

3. **`agent/`** — LLM-powered diagnostic agent. Triggered when RUL < 20% or sensor exceeds 3σ threshold. Collects recent 60-step data, sends to OpenAI-compatible API, returns structured fault report (root cause, action steps, spare parts list). Falls back to template-based reports if API unavailable.

4. **`dashboard/`** — FastAPI backend + HTML/ECharts frontend. Industrial HMI dark theme. REST API serves device status, telemetry, RUL trends, alerts, and AI diagnosis reports. Frontend polls every 3 seconds.

5. **`docs/`** — Design specs, README, interview script (STAR method with 祥发纺织/华为场景标注).

## Key Design Decisions

- Single-repo monolithic architecture (not microservices) for demo simplicity
- LLM integration via OpenAI-compatible API with configurable base_url/api_key/model_name
- Frontend is pure HTML/CSS/JS (no React/Vue) — easier to embed in interview demos
- Data is generated programmatically (not from files) — reproducible via seed
- All code has Chinese comments for interview presentation
