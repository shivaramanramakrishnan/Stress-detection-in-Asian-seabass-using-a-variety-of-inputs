"use client";
import { useState } from "react";

interface KeyFeatures {
  do_mg_per_l: number;
  ph: number;
  temp_c: number;
  nh3_mg_per_l: number;
  cortisol_ng_ml: number;
  microbiome_dysbiosis: number;
  fish_count: number;
  crowding_index: number;
  mortality_events: number;
  weather: string;
  system_type: string;
}

interface SamplePredictions {
  stress_score: number;
  high_stress: boolean;
  risk_level: "low" | "moderate" | "high";
  message: string;
}

interface SampleResponse {
  key_features: KeyFeatures;
  all_features: Record<string, any>;
  predictions: SamplePredictions;
}

interface ForecastStep {
  step: number;
  stress_score: number;
  high_stress: boolean;
  risk_level: "low" | "moderate" | "high";
  message: string;
  do_mg_per_l: number;
  temp_c: number;
  nh3_mg_per_l: number;
}

interface ForecastResponse {
  base_state: Record<string, any>;
  forecast_steps: number;
  trajectory: ForecastStep[];
}

interface LstmStep {
  step: number;
  stress_score: number;
  risk_level: "low" | "moderate" | "high";
}

interface LstmResponse {
  forecast_steps: number;
  trajectory: LstmStep[];
}

export default function Home() {
  const [loadingSample, setLoadingSample] = useState<boolean>(false);
  const [loadingForecast, setLoadingForecast] = useState<boolean>(false);
  const [loadingLstm, setLoadingLstm] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [lstmForecast, setLstmForecast] = useState<LstmResponse | null>(null);

  const fetchSample = async () => {
    setLoadingSample(true);
    setError("");
    try {
      const res = await fetch("http://localhost:8000/synthetic-sample");
      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const data: SampleResponse = await res.json();
      setSample(data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch sample from backend. Is the API running?");
    } finally {
      setLoadingSample(false);
    }
  };

  const fetchForecast = async () => {
    setLoadingForecast(true);
    setError("");
    try {
      const res = await fetch("http://localhost:8000/scenario-forecast?steps=12");
      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const data: ForecastResponse = await res.json();
      setForecast(data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch forecast from backend. Is the API running?");
    } finally {
      setLoadingForecast(false);
    }
  };

  const fetchLstmForecast = async () => {
    setLoadingLstm(true);
    setError("");
    try {
      const res = await fetch("http://localhost:8000/lstm-forecast?steps=24");
      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const data: LstmResponse = await res.json();
      setLstmForecast(data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch LSTM forecast. Is the API running and LSTM trained?");
    } finally {
      setLoadingLstm(false);
    }
  };

  const riskColor = (level: string) => {
    if (level === "high") return "#ef4444";
    if (level === "moderate") return "#f97316";
    return "#22c55e";
  };

  const renderRiskBadge = (level: string) => (
    <span
      style={{
        display: "inline-block",
        padding: "0.25rem 0.75rem",
        borderRadius: "999px",
        fontSize: "0.875rem",
        fontWeight: 600,
        backgroundColor: riskColor(level),
        color: "#020617",
      }}
    >
      {level === "low" ? "Low stress" : level === "moderate" ? "Moderate stress" : "High stress"}
    </span>
  );

  const panel = {
    padding: "1rem",
    borderRadius: "10px",
    backgroundColor: "#0b1120",
    border: "1px solid #1f2937",
  };

  const thStyle: React.CSSProperties = {
    padding: "0.5rem 0.75rem",
    textAlign: "left",
    color: "#9ca3af",
    fontWeight: 600,
  };

  const tdStyle: React.CSSProperties = {
    padding: "0.45rem 0.75rem",
    color: "#e5e7eb",
  };

  return (
    <main
      style={{
        minHeight: "100vh",
        padding: "2rem",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        backgroundColor: "#020617",
      }}
    >
      <div
        style={{
          maxWidth: "960px",
          margin: "0 auto",
          backgroundColor: "#020617",
          borderRadius: "16px",
          padding: "2rem",
          border: "1px solid #1f2937",
          boxShadow: "0 20px 40px rgba(15,23,42,0.75)",
        }}
      >
        {/* Header */}
        <h1 style={{ fontSize: "1.8rem", fontWeight: 700, marginBottom: "0.5rem", color: "#e5e7eb" }}>
          AI-enabled Stress Early Warning (Mock Farm)
        </h1>
        <p style={{ color: "#9ca3af", marginBottom: "1.5rem" }}>
          This module simulates Asian seabass culture conditions, fuses environmental, behavioural,
          biological, and management signals, and estimates a stress risk level from synthetic data
          using a Python backend. It also provides scenario-based and LSTM-based risk trajectories.
        </p>

        {/* Buttons */}
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          <button
            onClick={fetchSample}
            disabled={loadingSample}
            style={{
              padding: "0.75rem 1.5rem",
              borderRadius: "999px",
              border: "1px solid #0ea5e9",
              fontWeight: 600,
              cursor: loadingSample ? "not-allowed" : "pointer",
              backgroundColor: loadingSample ? "#0ea5e9" : "#38bdf8",
              color: "#020617",
            }}
          >
            {loadingSample ? "Simulating..." : "Simulate new scenario"}
          </button>

          <button
            onClick={fetchForecast}
            disabled={loadingForecast}
            style={{
              padding: "0.75rem 1.5rem",
              borderRadius: "999px",
              border: "1px solid #22c55e",
              fontWeight: 600,
              cursor: loadingForecast ? "not-allowed" : "pointer",
              backgroundColor: loadingForecast ? "#16a34a" : "#22c55e",
              color: "#020617",
            }}
          >
            {loadingForecast ? "Computing..." : "Scenario forecast (12 steps)"}
          </button>

          <button
            onClick={fetchLstmForecast}
            disabled={loadingLstm}
            style={{
              padding: "0.75rem 1.5rem",
              borderRadius: "999px",
              border: "1px solid #a78bfa",
              fontWeight: 600,
              cursor: loadingLstm ? "not-allowed" : "pointer",
              backgroundColor: loadingLstm ? "#7c3aed" : "#a78bfa",
              color: "#020617",
            }}
          >
            {loadingLstm ? "Running LSTM..." : "LSTM trajectory (24 steps)"}
          </button>
        </div>

        {error && (
          <p style={{ marginTop: "0.5rem", color: "#ef4444", fontWeight: 500 }}>{error}</p>
        )}

        {/* ── Current Scenario ── */}
        {sample && (
          <div style={{ marginTop: "2rem" }}>
            {/* Summary bar */}
            <section
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "1.5rem",
                ...panel,
              }}
            >
              <div>
                <h2 style={{ fontSize: "1.4rem", fontWeight: 600, marginBottom: "0.25rem", color: "#e5e7eb" }}>
                  Stress Intelligence Summary
                </h2>
                <p style={{ color: "#9ca3af", maxWidth: "540px" }}>
                  Stress score is derived from water chemistry, behaviour, biological markers,
                  environmental context, and lifecycle signals.
                </p>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: "2rem", fontWeight: 700, color: "#e5e7eb" }}>
                  {sample.predictions.stress_score.toFixed(2)}
                </div>
                <div style={{ marginTop: "0.5rem" }}>
                  {renderRiskBadge(sample.predictions.risk_level)}
                </div>
              </div>
            </section>

            {/* 4-panel grid */}
            <section
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "1.5rem",
                marginBottom: "1.5rem",
              }}
            >
              <div style={panel}>
                <h3 style={{ fontWeight: 600, marginBottom: "0.5rem", color: "#e5e7eb" }}>
                  Environmental & Water Chemistry
                </h3>
                <ul style={{ margin: 0, paddingLeft: "1.1rem", color: "#9ca3af" }}>
                  <li>DO: {sample.key_features.do_mg_per_l.toFixed(2)} mg/L</li>
                  <li>pH: {sample.key_features.ph.toFixed(2)}</li>
                  <li>Temperature: {sample.key_features.temp_c.toFixed(1)} °C</li>
                  <li>Ammonia (NH₃): {sample.key_features.nh3_mg_per_l.toFixed(3)} mg/L</li>
                  <li>Weather: {sample.key_features.weather}</li>
                  <li>System: {sample.key_features.system_type}</li>
                </ul>
              </div>

              <div style={panel}>
                <h3 style={{ fontWeight: 600, marginBottom: "0.5rem", color: "#e5e7eb" }}>
                  Behavioural & Biomass
                </h3>
                <ul style={{ margin: 0, paddingLeft: "1.1rem", color: "#9ca3af" }}>
                  <li>Fish count: {sample.key_features.fish_count.toFixed(0)}</li>
                  <li>Crowding index: {sample.key_features.crowding_index.toFixed(2)}</li>
                  <li>Mortality events: {sample.key_features.mortality_events.toFixed(1)}</li>
                </ul>
              </div>

              <div style={panel}>
                <h3 style={{ fontWeight: 600, marginBottom: "0.5rem", color: "#e5e7eb" }}>
                  Biological Markers
                </h3>
                <ul style={{ margin: 0, paddingLeft: "1.1rem", color: "#9ca3af" }}>
                  <li>Cortisol: {sample.key_features.cortisol_ng_ml.toFixed(1)} ng/mL</li>
                  <li>Microbiome dysbiosis: {sample.key_features.microbiome_dysbiosis.toFixed(2)}</li>
                </ul>
              </div>

              <div style={panel}>
                <h3 style={{ fontWeight: 600, marginBottom: "0.5rem", color: "#e5e7eb" }}>
                  Early Warning Insight
                </h3>
                <p style={{ margin: 0, color: "#f97316" }}>{sample.predictions.message}</p>
                <p style={{ marginTop: "0.5rem", color: "#9ca3af", fontSize: "0.9rem" }}>
                  High stress flag:{" "}
                  <strong style={{ color: "#e5e7eb" }}>
                    {sample.predictions.high_stress ? "Yes" : "No"}
                  </strong>
                </p>
              </div>
            </section>

            {/* Debug JSON */}
            <section>
              <h3 style={{ fontWeight: 600, marginBottom: "0.25rem", color: "#e5e7eb" }}>
                All Observables (debug view)
              </h3>
              <p style={{ color: "#9ca3af", fontSize: "0.85rem", marginBottom: "0.5rem" }}>
                Full synthetic feature vector used by the ML models.
              </p>
              <pre
                style={{
                  backgroundColor: "#020617",
                  color: "#e5e7eb",
                  padding: "1rem",
                  borderRadius: "8px",
                  fontSize: "0.8rem",
                  maxHeight: "260px",
                  overflow: "auto",
                  border: "1px solid #1f2937",
                }}
              >
{JSON.stringify(sample.all_features, null, 2)}
              </pre>
            </section>
          </div>
        )}

        {/* ── Scenario Forecast (RF) ── */}
        {forecast && (
          <div style={{ marginTop: "2.5rem" }}>
            <h2 style={{ fontSize: "1.4rem", fontWeight: 600, marginBottom: "0.5rem", color: "#e5e7eb" }}>
              Scenario Forecast — Worsening Water Quality (Random Forest)
            </h2>
            <p style={{ color: "#9ca3af", marginBottom: "0.75rem", maxWidth: "620px" }}>
              A simple rule-based scenario: DO drops, temperature and ammonia rise, crowding worsens.
              The Random Forest model estimates stress risk at each of 12 future steps.
            </p>
            <div style={{ borderRadius: "8px", border: "1px solid #1f2937", overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                <thead style={{ backgroundColor: "#0b1120", borderBottom: "1px solid #1f2937" }}>
                  <tr>
                    <th style={thStyle}>Step</th>
                    <th style={thStyle}>Stress score</th>
                    <th style={thStyle}>Risk level</th>
                    <th style={thStyle}>DO (mg/L)</th>
                    <th style={thStyle}>Temp (°C)</th>
                    <th style={thStyle}>NH₃ (mg/L)</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.trajectory.map((row) => (
                    <tr key={row.step} style={{ borderTop: "1px solid #1f2937" }}>
                      <td style={tdStyle}>{row.step}</td>
                      <td style={tdStyle}>{row.stress_score.toFixed(2)}</td>
                      <td style={{ ...tdStyle, color: riskColor(row.risk_level), fontWeight: 600 }}>
                        {row.risk_level}
                      </td>
                      <td style={tdStyle}>{row.do_mg_per_l.toFixed(2)}</td>
                      <td style={tdStyle}>{row.temp_c.toFixed(2)}</td>
                      <td style={tdStyle}>{row.nh3_mg_per_l.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── LSTM Forecast ── */}
        {lstmForecast && (
          <div style={{ marginTop: "2.5rem" }}>
            <h2 style={{ fontSize: "1.4rem", fontWeight: 600, marginBottom: "0.5rem", color: "#e5e7eb" }}>
              LSTM-based Risk Trajectory (24 steps)
            </h2>
            <p style={{ color: "#9ca3af", marginBottom: "0.75rem", maxWidth: "620px" }}>
              A trained LSTM model predicts stress scores across 24 sequential timesteps, learning
              temporal patterns from synthetic sensor sequences rather than applying a fixed rule.
              This demonstrates time-aware early warning, where hidden signal trends can be detected
              before they become visible.
            </p>
            <div style={{ borderRadius: "8px", border: "1px solid #1f2937", overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                <thead style={{ backgroundColor: "#0b1120", borderBottom: "1px solid #1f2937" }}>
                  <tr>
                    <th style={thStyle}>Step</th>
                    <th style={thStyle}>Stress score</th>
                    <th style={thStyle}>Risk level</th>
                  </tr>
                </thead>
                <tbody>
                  {lstmForecast.trajectory.map((row) => (
                    <tr key={row.step} style={{ borderTop: "1px solid #1f2937" }}>
                      <td style={tdStyle}>{row.step}</td>
                      <td style={tdStyle}>{row.stress_score.toFixed(2)}</td>
                      <td style={{ ...tdStyle, color: riskColor(row.risk_level), fontWeight: 600 }}>
                        {row.risk_level}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p style={{ color: "#6b7280", fontSize: "0.8rem", marginTop: "0.5rem" }}>
              Each step represents one simulated sensor reading window. The LSTM captures temporal
              dependencies across the full sequence, unlike the single-timestep Random Forest.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
