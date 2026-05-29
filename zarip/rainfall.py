import numpy as np
from scipy.stats import lognorm, norm

from .config import PROVINCE_COORDINATES, generate_spatial_correlation


class LognormEVTDist:
    def __init__(self, region: str, enable_evt: bool = True):
        self.region = region
        self.log_mu = None
        self.log_sigma = None
        self.u_threshold = None
        self.xi = -0.25
        self.sigma_evt = 45.0
        self.p_evt = 0.05
        self.enable_evt = enable_evt

    def fit(self, data: np.ndarray):
        log_specs = {
            'Region_I': (6.851, 0.147), 'Region_II': (6.678, 0.162),
            'Region_III': (6.545, 0.171), 'Region_IV': (6.296, 0.198), 'Region_V': (6.099, 0.221)
        }[self.region]
        self.log_mu, self.log_sigma = log_specs
        self.u_threshold = lognorm.ppf(0.10, s=self.log_sigma, scale=np.exp(self.log_mu))

    def rvs_from_u(self, u_arr: np.ndarray) -> np.ndarray:
        r_sim = np.zeros_like(u_arr)

        if self.enable_evt and self.p_evt > 0.0:
            tail_mask = u_arr < self.p_evt
            if np.any(tail_mask):
                u_scaled = u_arr[tail_mask] / self.p_evt
                u_scaled = np.clip(u_scaled, 1e-9, 1.0 - 1e-9)
                deficits = (self.sigma_evt / self.xi) * ((1.0 - u_scaled) ** (-self.xi) - 1.0)
                r_sim[tail_mask] = np.clip(self.u_threshold - deficits, 0.0, None)
        else:
            tail_mask = np.zeros_like(u_arr, dtype=bool)

        body_mask = ~tail_mask
        if np.any(body_mask):
            r_sim[body_mask] = lognorm.ppf(u_arr[body_mask], s=self.log_sigma, scale=np.exp(self.log_mu))

        return r_sim


class RainfallEVTModel:
    def __init__(self, use_evt: bool = True):
        self.regions = list(PROVINCE_COORDINATES.keys())
        self.use_evt = use_evt
        self.distributions = {reg: LognormEVTDist(reg, enable_evt=self.use_evt) for reg in self.regions}
        self.copula_corr = None

    def fit(self, df):
        for reg in self.regions:
            reg_values = df[df['region'] == reg]['rainfall'].values
            self.distributions[reg].fit(reg_values)
        self.copula_corr = generate_spatial_correlation(decay_parameter=200.0)
        for reg in self.regions:
            self.distributions[reg].enable_evt = self.use_evt

    def simulate(self, n_simulations: int = 100000, seed: int = 42) -> dict:
        np.random.seed(seed)
        L = np.linalg.cholesky(self.copula_corr)
        z = np.random.normal(0, 1, size=(n_simulations, len(self.regions)))
        z_corr = z @ L.T
        u = norm.cdf(z_corr)
        return {reg: self.distributions[reg].rvs_from_u(u[:, idx]) for idx, reg in enumerate(self.regions)}
