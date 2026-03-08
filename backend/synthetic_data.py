# synthetic_data.py

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def sample(self, size: int) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=size)

    def sample_int(self, size: int) -> np.ndarray:
        return np.random.randint(int(self.low), int(self.high) + 1, size=size)


@dataclass(frozen=True)
class EnvConfig:
    do: Range = Range(4.0, 8.0)
    ph: Range = Range(6.8, 8.2)
    nh3: Range = Range(0.0, 1.5)
    no2: Range = Range(0.0, 1.0)
    co2: Range = Range(0.0, 25.0)
    orp: Range = Range(150.0, 350.0)
    temp: Range = Range(25.0, 34.0)
    salinity: Range = Range(15.0, 35.0)
    light_lux: Range = Range(0.0, 10000.0)          # NEW: ambient light level


@dataclass(frozen=True)
class BehaviourConfig:
    swimming_distance: Range = Range(10.0, 200.0)
    tailbeat_freq: Range = Range(0.5, 5.0)
    crowding_index: Range = Range(0.1, 1.5)
    biomass_kg: Range = Range(50.0, 2000.0)
    fish_count: Range = Range(200.0, 20000.0)        # will be int


@dataclass(frozen=True)
class BiologicalConfig:
    cortisol: Range = Range(10.0, 200.0)
    microbiome_dysbiosis: Range = Range(0.0, 1.0)
    feeding_grams: Range = Range(0.0, 300.0)
    feeding_events: Range = Range(0.0, 5.0)         # will be int


@dataclass(frozen=True)
class HumanInfraConfig:
    human_presence_count: Range = Range(0, 5)        # int range
    human_dwell_minutes: Range = Range(0, 60)        # int range
    vibration_level: Range = Range(0.0, 1.0)
    ambient_noise_db: Range = Range(30.0, 100.0)


@dataclass(frozen=True)
class LifecycleConfig:
    mortality_events: Range = Range(0, 30)           # int range
    grading_events: Range = Range(0, 3)              # int range
    awg_grams: Range = Range(50.0, 1200.0)
    stocking_density: Range = Range(5.0, 50.0)


@dataclass
class SyntheticConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    behav: BehaviourConfig = field(default_factory=BehaviourConfig)
    bio: BiologicalConfig = field(default_factory=BiologicalConfig)
    human: HumanInfraConfig = field(default_factory=HumanInfraConfig)
    life: LifecycleConfig = field(default_factory=LifecycleConfig)
    n_samples: int = 10_000
    random_state: Optional[int] = 42


# Column order used by LSTM (numeric features only, fixed order)
# light_lux added at end to preserve backward compat with existing indexes
NUMERIC_COLS = [
    "do_mg_per_l", "ph", "nh3_mg_per_l", "no2_mg_per_l", "co2_mg_per_l",
    "orp_mv", "temp_c", "salinity_psu", "swimming_distance_m", "tailbeat_hz",
    "crowding_index", "biomass_kg", "fish_count", "cortisol_ng_ml",
    "microbiome_dysbiosis", "feeding_grams", "feeding_events",
    "human_presence_count", "human_dwell_minutes", "vibration_level",
    "ambient_noise_db", "mortality_events", "grading_events",
    "awg_grams", "stocking_density", "light_lux",               # NEW
]


# -----------------------------
# Synthetic data generator
# -----------------------------

class AquacultureSyntheticGenerator:
    def __init__(self, config: SyntheticConfig = SyntheticConfig()):
        self.config = config
        if config.random_state is not None:
            np.random.seed(config.random_state)

    def _sample_categorical(self, choices, probs, size):
        return np.random.choice(choices, p=probs, size=size)

    def generate(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        if n_samples is None:
            n_samples = self.config.n_samples

        env  = self.config.env
        beh  = self.config.behav
        bio  = self.config.bio
        hum  = self.config.human
        life = self.config.life

        # Environmental (floats)
        do       = env.do.sample(n_samples)
        ph       = env.ph.sample(n_samples)
        nh3      = env.nh3.sample(n_samples)
        no2      = env.no2.sample(n_samples)
        co2      = env.co2.sample(n_samples)
        orp      = env.orp.sample(n_samples)
        temp     = env.temp.sample(n_samples)
        salinity = env.salinity.sample(n_samples)
        light_lux = env.light_lux.sample(n_samples)              # NEW

        # Behavioural (fish_count → int)
        swimming_distance = beh.swimming_distance.sample(n_samples)
        tailbeat_freq     = beh.tailbeat_freq.sample(n_samples)
        crowding_index    = beh.crowding_index.sample(n_samples)
        biomass_kg        = beh.biomass_kg.sample(n_samples)
        fish_count        = beh.fish_count.sample_int(n_samples) # INT

        # Biological (feeding_events → int)
        cortisol              = bio.cortisol.sample(n_samples)
        microbiome_dysbiosis  = bio.microbiome_dysbiosis.sample(n_samples)
        feeding_grams         = bio.feeding_grams.sample(n_samples)
        feeding_events        = bio.feeding_events.sample_int(n_samples) # INT

        # Human / infra (presence count + dwell minutes → int)
        human_presence_count = hum.human_presence_count.sample_int(n_samples) # INT
        human_dwell_minutes  = hum.human_dwell_minutes.sample_int(n_samples)  # INT
        vibration_level      = hum.vibration_level.sample(n_samples)
        ambient_noise_db     = hum.ambient_noise_db.sample(n_samples)

        # Lifecycle (mortality + grading → int)
        mortality_events = life.mortality_events.sample_int(n_samples) # INT
        grading_events   = life.grading_events.sample_int(n_samples)   # INT
        awg_grams        = life.awg_grams.sample(n_samples)
        stocking_density = life.stocking_density.sample(n_samples)

        # Categorical
        system_type = self._sample_categorical(
            ["flow_through", "RAS"], [0.4, 0.6], n_samples
        )
        weather = self._sample_categorical(
            ["sunny", "cloudy", "rainy", "stormy"], [0.4, 0.3, 0.2, 0.1], n_samples
        )

        # -----------------------------
        # Derive latent stress score
        # -----------------------------
        stress = np.zeros(n_samples)

        stress += np.clip(4.0 - do, 0, 4) * 0.15
        stress += np.clip(nh3 - 0.5, 0, None) * 0.12
        stress += np.clip(no2 - 0.3, 0, None) * 0.08
        stress += np.clip(temp - 30.0, 0, None) * 0.08
        stress += np.clip(25.0 - temp, 0, None) * 0.05
        stress += np.clip(abs(ph - 7.5) - 0.3, 0, None) * 0.08
        stress += np.clip(crowding_index - 0.8, 0, None) * 0.2
        stress += np.clip(swimming_distance - 150.0, 0, None) * 0.002
        stress += np.clip(60.0 - swimming_distance, 0, None) * 0.002
        stress += np.clip(cortisol - 80.0, 0, None) * 0.01
        stress += microbiome_dysbiosis * 0.4
        stress += np.clip(ambient_noise_db - 70.0, 0, None) * 0.01
        stress += human_presence_count * 0.03
        stress += vibration_level * 0.2
        stress += mortality_events * 0.02
        stress += grading_events * 0.05
        stress += np.clip(stocking_density - 25.0, 0, None) * 0.03
        stress += np.where(weather == "stormy", 0.4, 0.0)
        stress += np.where(weather == "rainy", 0.15, 0.0)
        stress += np.where(system_type == "flow_through", 0.1, 0.0)
        # Light: very low light (night disturbance) adds mild stress
        stress += np.clip(500.0 - light_lux, 0, None) * 0.00005  # NEW

        # Normalize and add noise
        stress = stress - stress.min()
        if stress.max() > 0:
            stress = stress / stress.max()
        noise = np.random.normal(0, 0.05, size=n_samples)
        stress = np.clip(stress + noise, 0.0, 1.0)

        high_stress = (stress > 0.7).astype(int)

        df = pd.DataFrame(
            {
                "do_mg_per_l":            do,
                "ph":                     ph,
                "nh3_mg_per_l":           nh3,
                "no2_mg_per_l":           no2,
                "co2_mg_per_l":           co2,
                "orp_mv":                 orp,
                "temp_c":                 temp,
                "salinity_psu":           salinity,
                "swimming_distance_m":    swimming_distance,
                "tailbeat_hz":            tailbeat_freq,
                "crowding_index":         crowding_index,
                "biomass_kg":             biomass_kg,
                "fish_count":             fish_count,             # int
                "cortisol_ng_ml":         cortisol,
                "microbiome_dysbiosis":   microbiome_dysbiosis,
                "feeding_grams":          feeding_grams,
                "feeding_events":         feeding_events,         # int
                "human_presence_count":   human_presence_count,   # int
                "human_dwell_minutes":    human_dwell_minutes,    # int
                "vibration_level":        vibration_level,
                "ambient_noise_db":       ambient_noise_db,
                "mortality_events":       mortality_events,       # int
                "grading_events":         grading_events,         # int
                "awg_grams":              awg_grams,
                "stocking_density":       stocking_density,
                "light_lux":              light_lux,              # NEW
                "system_type":            system_type,
                "weather":                weather,
                "stress_score":           stress,
                "high_stress":            high_stress,
            }
        )

        return df

    def generate_sequences(
        self,
        n_sequences: int = 2000,
        timesteps: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate (X, y) sequences for LSTM training.
        X shape: (n_sequences, timesteps, n_numeric_features)
        y shape: (n_sequences, timesteps) — stress score at each step
        """
        X_all = []
        y_all = []

        IDX_DO       = NUMERIC_COLS.index("do_mg_per_l")
        IDX_NH3      = NUMERIC_COLS.index("nh3_mg_per_l")
        IDX_TEMP     = NUMERIC_COLS.index("temp_c")
        IDX_CROWDING = NUMERIC_COLS.index("crowding_index")
        IDX_CORTISOL = NUMERIC_COLS.index("cortisol_ng_ml")
        IDX_MICRO    = NUMERIC_COLS.index("microbiome_dysbiosis")
        IDX_HPC      = NUMERIC_COLS.index("human_presence_count")
        IDX_HDM      = NUMERIC_COLS.index("human_dwell_minutes")
        IDX_MORT     = NUMERIC_COLS.index("mortality_events")
        IDX_GRADE    = NUMERIC_COLS.index("grading_events")
        IDX_FISH     = NUMERIC_COLS.index("fish_count")
        IDX_FEED_EV  = NUMERIC_COLS.index("feeding_events")
        IDX_LIGHT    = NUMERIC_COLS.index("light_lux")             # NEW

        for _ in range(n_sequences):
            df_base  = self.generate(1)
            base_row = df_base[NUMERIC_COLS].values[0].astype(float).copy()

            seq_X = []
            seq_y = []

            for _ in range(timesteps):
                drift    = np.random.normal(0, 0.02, size=base_row.shape)
                base_row = base_row + drift

                # Float clips
                base_row[IDX_DO]       = np.clip(base_row[IDX_DO],       1.0,   12.0)
                base_row[IDX_NH3]      = np.clip(base_row[IDX_NH3],      0.0,    3.0)
                base_row[IDX_TEMP]     = np.clip(base_row[IDX_TEMP],    20.0,   40.0)
                base_row[IDX_CROWDING] = np.clip(base_row[IDX_CROWDING], 0.0,    3.0)
                base_row[IDX_CORTISOL] = np.clip(base_row[IDX_CORTISOL], 0.0,  300.0)
                base_row[IDX_MICRO]    = np.clip(base_row[IDX_MICRO],    0.0,    1.0)
                base_row[IDX_LIGHT]    = np.clip(base_row[IDX_LIGHT],    0.0, 10000.0) # NEW

                # Integer clips (round after drift)
                base_row[IDX_HPC]     = np.clip(round(base_row[IDX_HPC]),     0,  10)
                base_row[IDX_HDM]     = np.clip(round(base_row[IDX_HDM]),     0, 120)
                base_row[IDX_MORT]    = np.clip(round(base_row[IDX_MORT]),    0,  50)
                base_row[IDX_GRADE]   = np.clip(round(base_row[IDX_GRADE]),   0,   5)
                base_row[IDX_FISH]    = np.clip(round(base_row[IDX_FISH]),  100, 30000)
                base_row[IDX_FEED_EV] = np.clip(round(base_row[IDX_FEED_EV]), 0,  10)

                seq_X.append(base_row.copy())

                # Stress score from key signals
                stress  = 0.0
                stress += np.clip(4.0 - base_row[IDX_DO], 0, 4) * 0.15
                stress += np.clip(base_row[IDX_NH3] - 0.5, 0, None) * 0.12
                stress += np.clip(base_row[IDX_TEMP] - 30.0, 0, None) * 0.08
                stress += np.clip(base_row[IDX_CROWDING] - 0.8, 0, None) * 0.2
                stress += np.clip(base_row[IDX_CORTISOL] - 80.0, 0, None) * 0.01
                stress += base_row[IDX_MICRO] * 0.4
                stress += np.clip(500.0 - base_row[IDX_LIGHT], 0, None) * 0.00005  # NEW
                stress  = float(np.clip(stress, 0.0, 1.0))
                seq_y.append(stress)

            X_all.append(seq_X)
            y_all.append(seq_y)

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        return X, y, NUMERIC_COLS


if __name__ == "__main__":
    cfg = SyntheticConfig(n_samples=5000, random_state=123)
    gen = AquacultureSyntheticGenerator(cfg)
    df  = gen.generate()
    print(df.head())
    print("\nDtype check for integer columns:")
    for col in ["fish_count", "feeding_events", "human_presence_count",
                "human_dwell_minutes", "mortality_events", "grading_events"]:
        print(f"  {col}: {df[col].dtype} — sample values: {df[col].values[:5]}")
    df.to_csv("aquaculture_synthetic.csv", index=False)
    print("\nSaved aquaculture_synthetic.csv")

    print("\nTesting generate_sequences...")
    X, y, cols = gen.generate_sequences(n_sequences=10, timesteps=24)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Feature count: {len(cols)}")
