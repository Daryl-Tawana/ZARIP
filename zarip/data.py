import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm

from .config import PROVINCE_COORDINATES
from .logger import logger


def generate_historical_series() -> pd.DataFrame:
    np.random.seed(42)
    regions = list(PROVINCE_COORDINATES.keys())
    years = list(range(1980, 2026))

    reg_specs = {
        'Region_I':   {'mean': 950.0, 'std': 140.0, 'y_max': 2.42, 'k': 0.0183, 'r0': 467.0},
        'Region_II':  {'mean': 800.0, 'std': 130.0, 'y_max': 2.42, 'k': 0.0183, 'r0': 467.0},
        'Region_III': {'mean': 700.0, 'std': 120.0, 'y_max': 2.42, 'k': 0.0183, 'r0': 467.0},
        'Region_IV':  {'mean': 550.0, 'std': 110.0, 'y_max': 2.42, 'k': 0.0183, 'r0': 467.0},
        'Region_V':   {'mean': 450.0, 'std': 100.0, 'y_max': 2.42, 'k': 0.0183, 'r0': 467.0}
    }

    from .config import generate_spatial_correlation

    corr_matrix = generate_spatial_correlation(decay_parameter=200.0)
    L = np.linalg.cholesky(corr_matrix)

    records = []
    for yr in years:
        z_ind = np.random.normal(0, 1, size=len(regions))
        z_corr = L @ z_ind
        u_corr = norm.cdf(z_corr)

        for idx, reg in enumerate(regions):
            spec = reg_specs[reg]
            log_mu = {
                'Region_I': 6.851, 'Region_II': 6.678, 'Region_III': 6.545,
                'Region_IV': 6.296, 'Region_V': 6.099
            }[reg]
            log_sigma = {
                'Region_I': 0.147, 'Region_II': 0.162, 'Region_III': 0.171,
                'Region_IV': 0.198, 'Region_V': 0.221
            }[reg]

            rain = lognorm.ppf(u_corr[idx], s=log_sigma, scale=np.exp(log_mu))
            rain = max(50.0, rain)

            y_base = spec['y_max'] / (1.0 + np.exp(-spec['k'] * (rain - spec['r0'])))
            epsilon = np.random.normal(0, 0.15)
            y_val = np.clip(y_base + epsilon, 0.0, spec['y_max'])

            ndvi = 0.15 + 0.50 / (1.0 + np.exp(-0.01 * (rain - 450.0))) + np.random.normal(0, 0.02)
            ndvi = np.clip(ndvi, 0.1, 0.8)

            records.append({
                'region': reg,
                'year': yr,
                'rainfall': rain,
                'yield': y_val,
                'ndvi': ndvi
            })

    df = pd.DataFrame(records)
    for col in ['rainfall', 'yield', 'ndvi']:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    return df


class DataProcessor:
    def __init__(self):
        self.data = None
        self.quality_report = {}

    def clean_and_harmonize(self, raw_df: pd.DataFrame = None) -> pd.DataFrame:
        raw_df = raw_df if raw_df is not None else generate_historical_series()
        df = raw_df.copy()
        df['ag_year'] = df['year'].astype(int)

        self.quality_report['records_count'] = len(df)
        self.quality_report['missing_before'] = df.isnull().sum().to_dict()

        df = df.sort_values(by=['region', 'ag_year']).reset_index(drop=True)
        for col in ['rainfall', 'yield', 'ndvi']:
            if col in df.columns:
                df[col] = df.groupby('region')[col].transform(lambda x: x.interpolate(method='linear').bfill().ffill())

        self.quality_report['missing_after'] = df.isnull().sum().to_dict()

        if 'yield' in df.columns:
            df['yield_winsorized'] = df['yield']
            for reg in df['region'].unique():
                reg_mask = df['region'] == reg
                cutoff = df.loc[reg_mask, 'yield'].quantile(0.99)
                df.loc[reg_mask, 'yield_winsorized'] = df.loc[reg_mask, 'yield'].clip(upper=cutoff)

        self.data = df
        return df

    def save_quality_report(self, output_dir: str):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "zarip_data_quality_report.json", "w") as f:
            json.dump(self.quality_report, f, indent=4)
        logger.info(f"Data quality report successfully written to {path / 'zarip_data_quality_report.json'}")


class DataLoader:
    REQUIRED_COLS = {'region', 'year', 'rainfall', 'yield'}

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.regions: list[str] = []

    def load_default(self) -> None:
        self.df = generate_historical_series()
        self.regions = sorted(self.df['region'].unique().tolist())

    def load_csv(self, path: str) -> None:
        df = pd.read_csv(path)
        if 'rainfall_mm' in df.columns:
            df['rainfall'] = df['rainfall_mm']
        if 'yield_t_ha' in df.columns:
            df['yield'] = df['yield_t_ha']

        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        if df[list(self.REQUIRED_COLS)].isnull().values.any():
            raise ValueError("CSV contains missing values for required columns.")

        self.df = df.copy()
        self.regions = sorted(self.df['region'].unique().tolist())
