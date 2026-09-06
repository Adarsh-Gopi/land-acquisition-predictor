# 🏗️ Predictive Analytics System for Early Detection of Land Acquisition Delays

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Node.js](https://img.shields.io/badge/Node.js-Express-339933?logo=node.js)](https://nodejs.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas%2FLocal-47A248?logo=mongodb)](https://mongodb.com)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Dual--Engine-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **SIH (Smart India Hackathon) Project** — Land acquisition is one of the most critical and time-sensitive phases of infrastructure development. Delays in acquiring land significantly impact the execution of national and state-level projects. This system predicts the success probability and risk level of land acquisition cases — from **Low** to **High** — enabling early intervention.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [ML Model](#-ml-model--performance)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)

---

## 🎯 Problem Statement

Infrastructure projects in India frequently face delays due to unresolved land acquisition cases. Factors like title disputes, court cases, compensation funding gaps, and fragmented land registries across multiple villages create cascading delays. Early detection of high-risk cases allows administrators to prioritize intervention and prevent project overruns.

---

## 💡 Solution Overview

A full-stack web application that:

1. Accepts land acquisition case parameters via a web form
2. Sends the data to a **FastAPI ML microservice**
3. Runs a **Dual-Engine XGBoost model** to predict:
   - **Delay Probability** (continuous 0–100%)
   - **Risk Level** (Low / Medium / High)
   - **Top Risk Factors** driving the delay
   - **Recommended Preventive Actions** for administrators
4. Saves results to **MongoDB** for historical tracking
5. Displays a detailed risk assessment dashboard

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User (Browser)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP (Port 3000)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Express.js Frontend Server (Node.js)                   │
│  • EJS templating + Bootstrap UI                                │
│  • Routes: /, /predict, /history, /history/:id                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP POST /predict (Port 8000)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          FastAPI ML Microservice (Python)                        │
│  • Feature Engineering                                          │
│  • XGBoost Dual-Engine Inference                                │
│  • Risk Factor & Action Generation                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
  ┌──────────────────┐     ┌──────────────────────┐
  │  MongoDB (Local) │     │ In-Memory Fallback    │
  │  land_acquisition│     │ (if MongoDB offline)  │
  │  _db             │     └──────────────────────┘
  └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML API** | Python · FastAPI · XGBoost · scikit-learn · pandas · joblib |
| **Web Backend** | Node.js · Express.js v5 |
| **Frontend** | EJS · EJS-Mate · Bootstrap |
| **Database** | MongoDB (Mongoose) with in-memory fallback |
| **Dev Tools** | concurrently (runs both servers simultaneously) |

---

## 🤖 ML Model & Performance

The model uses a **Dual-Engine XGBoost** architecture:

| Engine | Task | Performance |
|---|---|---|
| XGBoost Regressor | Continuous delay probability (0.0 – 1.0) | R² = **0.9815**, MAE = **2.17%** |
| XGBoost Classifier | 3-Class risk tier (Low / Medium / High) | Accuracy = **93.00%** |

**Engineered Features:**

| Feature | Description |
|---|---|
| `dispute_score` | Sum of title dispute + court case + objection flags |
| `severe_legal_risk` | 1 if both court case AND title dispute are active |
| `area_per_owner` | Land area divided by number of affected owners |
| `milestone_avg` | Average of 4 acquisition milestone percentages |
| `possession_lag` | Gap between awards completed and possession completed |
| `disbursal_ratio` | Compensation disbursed vs. awards completed ratio |
| `unfunded_amount` | Estimated unfunded compensation in INR Lakhs |

**Training Data:** 2,000 cases · **Test Set:** 500 cases

---

## ✨ Features

- 📊 **Dashboard** — Live stats showing total cases, high/medium/low risk breakdown
- 📝 **Case Assessment Form** — Submit new land acquisition cases with 18+ parameters
- 🔮 **Risk Prediction** — Instant delay probability and risk tier classification
- ⚠️ **Risk Factor Analysis** — Specific reasons why a case is high-risk
- 💊 **Preventive Actions** — Actionable recommendations for administrators
- 📁 **Case History** — Browse and review all past assessments
- 🔌 **REST API** — Standalone FastAPI with Swagger docs at `/docs`
- 💾 **Persistent Storage** — MongoDB with automatic in-memory fallback

---

## ✅ Prerequisites

- [Node.js](https://nodejs.org/) v18+
- [Python](https://python.org/) 3.10+
- [MongoDB](https://www.mongodb.com/try/download/community) (optional — app runs without it)
- `pip` and `npm` installed

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Adarsh-Gopi/land-acquisition-predictor.git
cd land-acquisition-predictor
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Install Python dependencies

```bash
pip install fastapi uvicorn[standard] xgboost scikit-learn pandas numpy joblib pydantic
```

### 4. (Optional) Configure MongoDB

By default, the app connects to `mongodb://127.0.0.1:27017/land_acquisition_db`.

To use a custom URI, set the environment variable:

```bash
# Windows
set MONGO_URI=mongodb://your-connection-string

# Linux/macOS
export MONGO_URI=mongodb://your-connection-string
```

If MongoDB is not running, the app automatically falls back to in-memory storage.

---

## 🚀 Running the Application

### Option 1 — Run both servers simultaneously (recommended)

```bash
npm run dev
```

This starts:
- **FastAPI ML API** → `http://localhost:8000`
- **Express Web App** → `http://localhost:3000`

### Option 2 — Run servers individually

```bash
# Terminal 1 — FastAPI ML Server
npm run python
# or: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Express Web Server
npm start
# or: node app.js
```

### Option 3 — Windows batch file

```bash
start_unified.bat
```

Open your browser at **[http://localhost:3000](http://localhost:3000)**

---

## 🔌 API Reference

The FastAPI server exposes a REST API at `http://localhost:8000`. Interactive Swagger docs available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

### `GET /health`
Returns model status and accuracy summary.

### `GET /model/info`
Returns model architecture, feature list, and full metrics.

### `POST /predict`
Predict delay probability for a single case.

**Request body example:**
```json
{
  "case_id": "LA-2026-UP-001",
  "state": "Uttar Pradesh",
  "district_type": "Rural",
  "project_type": "Railway",
  "land_type": "Agricultural",
  "notification_stage": "3D issued",
  "land_area_hectares": 3.32,
  "affected_owner_count": 12,
  "is_multi_village": 1,
  "has_title_dispute": 1,
  "has_court_case": 0,
  "has_objection": 1,
  "compensation_estimate_inr_lakh": 250.0,
  "compensation_funding_available": 1,
  "days_since_case_opened": 240,
  "land_notified_percent": 80.0,
  "award_completed_percent": 60.0,
  "compensation_disbursed_percent": 45.0,
  "possession_completed_percent": 25.0
}
```

**Response:**
```json
{
  "case_id": "LA-2026-UP-001",
  "delay_probability": 0.7234,
  "delay_probability_percent": "72.3%",
  "risk_level": "High",
  "is_delayed_predicted": 1,
  "risk_distribution": { "low": 0.08, "medium": 0.19, "high": 0.73 },
  "top_risk_factors": ["Ownership title dispute among claimants", "..."],
  "recommended_preventive_actions": ["Deposit disputed share into escrow...", "..."],
  "processing_time_ms": 4.21
}
```

### `POST /predict/batch`
Batch predict risk for multiple cases at once.

---

## 📁 Project Structure

```
land-acquisition-predictor/
├── main.py                          # FastAPI ML microservice
├── app.js                           # Express.js web server
├── package.json                     # Node.js dependencies
├── start_unified.bat                # Windows quick-start script
│
├── config/
│   └── db.js                        # MongoDB connection
│
├── routes/
│   └── caseRoutes.js                # Express route handlers
│
├── database/
│   └── Case.js                      # Mongoose schema
│
├── services/
│   └── mlService.js                 # FastAPI client service
│
├── models/
│   ├── land_acquisition_delay_prototype.joblib   # Trained ML pipeline
│   ├── prototype_model_metrics.json              # Model performance metrics
│   ├── prototype_land_acquisition_cases.csv      # Training dataset
│   └── src_code/
│       ├── train_prototype_model.py              # Model training script
│       └── prototype_model_experiment.ipynb      # Experiment notebook
│
├── views/
│   └── listing/                     # EJS templates
│
├── public/                          # Static assets (CSS, JS)
│
└── Arch. FlowChart.jpeg             # System architecture diagram
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
