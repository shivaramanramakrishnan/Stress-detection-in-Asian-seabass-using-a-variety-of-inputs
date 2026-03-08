# AI-Enabled Stress Early Warning System — Asian Seabass Aquaculture

A minimal but complete predictive stress intelligence module for detecting early stress signals in farmed Asian seabass (*Lates calcarifer*), built with a Python FastAPI backend and a Next.js TypeScript frontend.


---

##  Project Goal

Conventional aquaculture detects fish stress when visible symptoms appear. This system aims to identify stress earlier by fusing signals across five categories:

-  **Environmental / Water Chemistry** — DO, pH, NH₃, NO₂, CO₂, ORP, temperature, salinity, light level
-  **Behavioural** — swimming distance, tailbeat frequency, crowding index, biomass, fish count
-  **Biological Markers** — cortisol levels, microbiome dysbiosis index
-  **Environmental Context** — weather, ambient noise, vibration, human presence & exposure time
-  **Lifecycle / Farm Management** — mortality events, grading events, stocking density, AWG

---

## Tools I used
- Cursor IDE - to write and run codes
- Preplexity Research Mode - to research and read papers on the different signals that are combined to read about the fish health

---

## Architecture

```
backend/
  synthetic_data.py     # Multi-signal synthetic data generator + LSTM sequences
  train_models.py       # Random Forest regressor + classifier training
  train_lstm.py         # LSTM neural network training
  api.py                # FastAPI backend
  requirements.txt

frontend/
  app/
    page.tsx            # Next.js dashboard
```

```
Python Backend (FastAPI :8000)
  ├── /health
  ├── /stress-ontology       → signal category structure
  ├── /synthetic-sample      → Random Forest snapshot prediction
  ├── /scenario-forecast     → rule-based worsening scenario (RF)
  └── /lstm-forecast         → learned temporal trajectory (LSTM)
        ↕ HTTP
Next.js Frontend (:3000)
  ├── Simulate new scenario  
  ├── Scenario forecast      
  └── LSTM trajectory        
```

---

## AI Models

| Model | Type | Purpose |
|---|---|---|
| `RandomForestRegressor` | Classic ML | Predicts continuous stress score (0.0–1.0) from current sensor snapshot |
| `RandomForestClassifier` | Classic ML | Binary high-stress flag (yes/no alert) |
| `StressLSTM` | Deep Learning (PyTorch) | Learns temporal stress patterns across 24-step sensor sequences |

The **scenario forecast** applies a rule-based worsening drift (DO drops, temp and NH₃ rise) and re-runs the Random Forest at each step — simple but interpretable.

The **LSTM** captures temporal dependencies in sensor sequences, detecting hidden signal trends before they manifest as visible stress.

---

## Getting Started

### Prerequisites

- Python ≥ 3.9
- Node.js ≥ 18
- `pip` and `npm`

---

### Backend Setup

Open **Command Prompt** and navigate to the backend folder:

```bash
cd ai-aquaculture/backend
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install torch
```

---

### Train the Models (one-time)

**Step 1 — Random Forest:**
```bash
python train_models.py
```
Generates `aquaculture_synthetic.csv`, trains and saves:
- `stress_regressor.joblib`
- `stress_classifier.joblib`

**Step 2 — LSTM:**
```bash
python train_lstm.py
```
Trains on 2,000 synthetic sequences of 24 timesteps, saves:
- `stress_lstm.pt`
- `stress_lstm_meta.joblib`

---

### Start the Backend

```bash
uvicorn api:app --reload --port 8000
```

Verify at:
- `http://localhost:8000/health` → `{"status": "ok"}`
- `http://localhost:8000/synthetic-sample` → single scenario JSON
- `http://localhost:8000/scenario-forecast?steps=12` → 12-step RF forecast
- `http://localhost:8000/lstm-forecast?steps=24` → 24-step LSTM trajectory

---

### Frontend Setup

Open a new terminal:

```bash
cd ai-aquaculture/frontend
```

If first time:
```bash
npx create-next-app@latest .
```

Install and run:
```bash
npm install
npm run dev
```

Open `http://localhost:3000`

---

### Re-running After First Setup

**Terminal 1 — Backend:**
```bash
cd ai-aquaculture/backend
.venv\Scripts\activate.bat
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd ai-aquaculture/frontend
npm run dev
```

---

## Signal Coverage

| Signal | Included |
|---|---|
| Swimming metrics (distance, tailbeat) | ✅ |
| Crowding & biomass estimation | ✅ |
| DO, pH, NH₃, CO₂, ORP, salinity | ✅ |
| Temperature | ✅ |
| Light level (lux) | ✅ |
| Cortisol levels | ✅ |
| Microbiome dysbiosis index | ✅ |
| Weather context | ✅ |
| Human presence & dwell time | ✅ |
| Ambient noise & vibration | ✅ |
| Mortality & grading events | ✅ |
| Stocking density & AWG | ✅ |
| Visual/camera baselines | 🔄 Scalar proxies (future: YOLO/ByteTrack) |
| Audio spectrum analysis | 🔄 Scalar proxy via noise dB (future: hydrophone) |

---


## Future Work

- **Seed sharing** — connect all three endpoints to the same farm state for consistent numbers
- **Real sensor integration** — replace synthetic generator with a real time DB stream
- **LLM-augmented reasoning** — natural language risk narrative via api call with a pre running GPT or local Ollama model
- **Computer vision module** — YOLO/ByteTrack fish tracking to replace scalar proxies
- **Digital twin** — physics-based tank simulation with real-time sensor coupling

---

## Key Files

| File | Purpose |
|---|---|
| `synthetic_data.py` | Generates synthetic sensor rows and LSTM sequences |
| `train_models.py` | Trains Random Forest regressor and classifier |
| `train_lstm.py` | Trains PyTorch LSTM on 24-step sequences |
| `api.py` | FastAPI server with for backend data |
| `app/page.tsx` | Next.js dashboard |

---
