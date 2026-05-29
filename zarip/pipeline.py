import os

from .config import DEFAULT_REGIONAL_EXPOSURES
from .data import DataProcessor
from .insurance import YieldModel, InsuranceEngine, RiskAnalyzer, SensitivityAnalyzer
from .rainfall import RainfallEVTModel
from .reporting import PolicyReporter
from .logger import logger


def run_zarip_pipeline(output_dir: str = "output", n_simulations: int = 100000,
                       sum_insured_usd_per_ha: float = 400.0, premium_loading: float = 1.25,
                       trigger_pct: float = 25.0, exit_pct: float = 5.0,
                       use_evt: bool = True, raw_df=None,
                       seed: int = 42):
    logger.info("Starting ZARIP actuarial pipeline")
    os.makedirs(output_dir, exist_ok=True)

    regional_exposures = DEFAULT_REGIONAL_EXPOSURES
    processor = DataProcessor()
    cleaned_df = processor.clean_and_harmonize(raw_df)
    processor.save_quality_report(output_dir)

    rainfall_model = RainfallEVTModel(use_evt=use_evt)
    rainfall_model.fit(cleaned_df)

    sim_rain = rainfall_model.simulate(n_simulations=n_simulations, seed=seed)

    yield_model = YieldModel()
    ins_engine = InsuranceEngine(sum_insured=sum_insured_usd_per_ha)
    ins_engine.calibrate(cleaned_df, trigger_pct=trigger_pct, exit_pct=exit_pct)

    sim_payouts = {reg: ins_engine.calculate_payout(reg, sim_rain[reg]) for reg in regional_exposures.keys()}

    analyzer = RiskAnalyzer(exposures=regional_exposures,
                            sum_insured_per_ha=sum_insured_usd_per_ha,
                            premium_loading=premium_loading)
    portfolio_metrics, regional_df = analyzer.analyze(sim_payouts)

    sens_analyzer = SensitivityAnalyzer(processor, regional_exposures, sum_insured_usd_per_ha)
    sens_df = sens_analyzer.run_sweep(portfolio_metrics['VaR95_usd'])

    reporter = PolicyReporter(output_dir=output_dir)
    reporter.export_csvs(regional_df, portfolio_metrics, sens_df)
    reporter.generate_all_charts(cleaned_df, sim_rain, sim_payouts,
                                 regional_exposures, regional_df, sens_df,
                                 yield_model, ins_engine, rainfall_model)
    reporter.generate_briefing(portfolio_metrics, regional_df, sens_df)

    logger.info("Completed ZARIP actuarial pipeline")
    return portfolio_metrics, regional_df, sens_df, cleaned_df, sim_rain, sim_payouts, rainfall_model, ins_engine
