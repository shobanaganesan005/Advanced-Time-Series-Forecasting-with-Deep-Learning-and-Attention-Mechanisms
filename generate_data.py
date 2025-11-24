
import numpy as np
import pandas as pd


def generate_synthetic_series(
    n_timesteps: int = 1500,
    n_features: int = 3,
    seed: int = 42,
    freq: str = "D"
):
    """
    Generate a multivariate time series with:
    - global linear trend
    - multiple seasonalities
    - feature-specific noise
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_timesteps)

    # Global components
    trend = 0.005 * t
    season1 = 0.8 * np.sin(2 * np.pi * t / 24)      # daily-like
    season2 = 0.4 * np.sin(2 * np.pi * t / 168)     # weekly-like

    data = []
    for f in range(n_features):
        phase_shift = rng.uniform(0, 2 * np.pi)
        feature = (
            trend
            + season1 * (1 + 0.2 * f)
            + season2 * (1 + 0.1 * f)
            + 0.3 * rng.standard_normal(size=n_timesteps)
        )
        feature = feature + 0.2 * np.sin(2 * np.pi * t / 48 + phase_shift)
        data.append(feature)

    data = np.stack(data, axis=1)

    idx = pd.date_range(start="2020-01-01", periods=n_timesteps, freq=freq)
    cols = [f"feat_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, index=idx, columns=cols)
    return df


def main():
    df = generate_synthetic_series()
    path = "data.csv"
    df.to_csv(path)
    print(f"Saved dataset with shape {df.shape} to {path}")


if __name__ == "__main__":
    main()
