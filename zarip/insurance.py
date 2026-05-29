import numpy as np
import pandas as pd

from .data import DataProcessor


class YieldModel:
    def __init__(self):
        self.y_max = 2.42
        self.k = 0.0183
        self.r0 = 467.0
        self.sigma_basis = 0.15

    def simulate_yield(self, sim_rainfall: np.ndarray, basis_scale: float = 1.0) -> np.ndarray:
        y_base = self.y_max / (1.0 + np.exp(-self.k * (sim_rainfall - self.r0)))
        epsilon = np.random.normal(0, self.sigma_basis * basis_scale, size=len(sim_rainfall))
        return np.clip(y_base + epsilon, 0.0, self.y_max)


class InsuranceEngine:
    def __init__(self, sum_insured: float = 400.0):
        self.sum_insured = sum_insured
        self.triggers = {}

    def calibrate(self, df: pd.DataFrame, trigger_pct: float = 25.0, exit_pct: float = 5.0):
        for reg in df['region'].unique():
            reg_rain = df[df['region'] == reg]['rainfall'].values
            self.triggers[reg] = {
                'trigger': np.percentile(reg_rain, trigger_pct),
                'exit': np.percentile(reg_rain, exit_pct)
            }

    def calculate_payout(self, region: str, rain_sim: np.ndarray, trigger_override: float = None) -> np.ndarray:
        t = self.triggers[region]
        trig = trigger_override if trigger_override is not None else t['trigger']
        ex = t['exit']

        payouts = np.zeros_like(rain_sim)
        payouts[rain_sim <= ex] = self.sum_insured
        linear_mask = (rain_sim > ex) & (rain_sim < trig)
        payouts[linear_mask] = self.sum_insured * (trig - rain_sim[linear_mask]) / (trig - ex)
        return payouts


class RiskAnalyzer:
    def __init__(self, exposures: dict, sum_insured_per_ha: float, premium_loading: float = 1.25):
        self.exposures = exposures
        self.sum_insured_per_ha = sum_insured_per_ha
        self.premium_loading = premium_loading

    def analyze(self, sim_payouts: dict) -> tuple[dict, pd.DataFrame]:
        n_sim = len(next(iter(sim_payouts.values())))
        portfolio_losses = np.zeros(n_sim)
        regional_metrics = []

        for reg, payouts in sim_payouts.items():
            exp = self.exposures[reg]
            portfolio_losses += payouts * exp

            exp_payout = payouts.mean()
            var95 = np.percentile(payouts, 95)
            cvar95 = payouts[payouts >= var95].mean() if np.any(payouts >= var95) else var95

            regional_metrics.append({
                'region': reg,
                'exposure_ha': exp,
                'expected_loss_usd': exp_payout * exp,
                'pure_premium_rate_pct': (exp_payout / self.sum_insured_per_ha * 100) if self.sum_insured_per_ha > 0 else 0,
                'loaded_premium_rate_pct': (exp_payout * self.premium_loading / self.sum_insured_per_ha * 100) if self.sum_insured_per_ha > 0 else 0,
                'VaR95_usd': var95 * exp,
                'CVaR95_usd': cvar95 * exp
            })

        regional_df = pd.DataFrame(regional_metrics)
        expected_portfolio_loss = portfolio_losses.mean()
        loaded_premium = expected_portfolio_loss * self.premium_loading

        p_var95 = np.percentile(portfolio_losses, 95)
        p_var99 = np.percentile(portfolio_losses, 99)
        p_cvar95 = portfolio_losses[portfolio_losses >= p_var95].mean()
        p_cvar99 = portfolio_losses[portfolio_losses >= p_var99].mean()

        portfolio_metrics = {
            'total_exposure_ha': sum(self.exposures.values()),
            'total_sum_insured_usd': sum(r['exposure_ha'] * self.sum_insured_per_ha for r in regional_metrics),
            'expected_loss_usd': expected_portfolio_loss,
            'loaded_premium_usd': loaded_premium,
            'VaR95_usd': p_var95,
            'VaR99_usd': p_var99,
            'CVaR95_usd': p_cvar95,
            'CVaR99_usd': p_cvar99,
            'tail_ratio': (p_cvar95 / expected_portfolio_loss) if expected_portfolio_loss > 0 else 1.0,
            'contingent_liability_95_usd': max(0.0, p_var95 - loaded_premium),
            'contingent_liability_99_usd': max(0.0, p_var99 - loaded_premium),
            'contingent_liability_mean_usd': max(0.0, expected_portfolio_loss - loaded_premium),
            'prob_any_payout': float((portfolio_losses > 0).mean()),
            'max_loss_usd': float(portfolio_losses.max()),
            'std_loss_usd': float(portfolio_losses.std())
        }
        return portfolio_metrics, regional_df


class SensitivityAnalyzer:
    def __init__(self, data_proc: DataProcessor, exposures: dict, sum_insured: float):
        self.data_proc = data_proc
        self.exposures = exposures
        self.sum_insured = sum_insured

    def run_sweep(self, baseline_var95: float) -> pd.DataFrame:
        params = ['Rainfall Mean', 'Rainfall Std Dev', 'Basis Risk Std Dev', 'EVT Tail Probability', 'Yield Steepness k', 'Insurance Trigger Level']
        results = []
        for p in params:
            low_var = self._simulate_perturbation(p, 0.8)
            high_var = self._simulate_perturbation(p, 1.2)
            results.append({
                'Parameter': p,
                'Low_Var95_Pct': (low_var - baseline_var95) / baseline_var95 * 100,
                'High_Var95_Pct': (high_var - baseline_var95) / baseline_var95 * 100
            })
        return pd.DataFrame(results)

    def _simulate_perturbation(self, param: str, factor: float) -> float:
        df_mod = self.data_proc.data.copy()
        basis_scale = 1.0
        trigger_scale = 1.0

        if param == 'Rainfall Mean':
            df_mod['rainfall'] = df_mod['rainfall'] * factor
        elif param == 'Rainfall Std Dev':
            mean = df_mod.groupby('region')['rainfall'].transform('mean')
            df_mod['rainfall'] = mean + (df_mod['rainfall'] - mean) * factor
        elif param == 'Basis Risk Std Dev':
            basis_scale = factor
        elif param == 'Insurance Trigger Level':
            trigger_scale = factor

        from .rainfall import RainfallEVTModel

        n_sim = 20000
        rf = RainfallEVTModel()
        rf.fit(df_mod)
        sim_rain = rf.simulate(n_simulations=n_sim, seed=123)

        ins = InsuranceEngine(sum_insured=self.sum_insured)
        ins.calibrate(df_mod)

        p_loss = np.zeros(n_sim)
        for reg, exp in self.exposures.items():
            trig_override = ins.triggers[reg]['trigger'] * trigger_scale
            payouts = ins.calculate_payout(reg, sim_rain[reg], trigger_override=trig_override)
            p_loss += payouts * exp

        return np.percentile(p_loss, 95)
