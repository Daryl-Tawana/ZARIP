"""ZARIP desktop and command-line launcher."""

import argparse
import os

from zarip.data import DataLoader
from zarip.gui import ZARIPApp
from zarip.pipeline import run_zarip_pipeline
from zarip.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="ZARIP Zimbabwe Agricultural Risk Insurance Platform")
    parser.add_argument("--nogui", action="store_true", help="Run the actuarial pipeline in CLI mode without opening the desktop interface.")
    parser.add_argument("--csv", type=str, help="Path to a historical data CSV file.")
    parser.add_argument("--output-dir", default="output", help="Directory to write charts, CSV exports, and briefing text.")
    parser.add_argument("--iterations", type=int, default=100000, help="Number of Monte Carlo simulations to run.")
    parser.add_argument("--sum-insured", type=float, default=400.0, help="Sum insured per hectare in USD.")
    parser.add_argument("--premium-loading", type=float, default=1.25, help="Premium loading multiplier.")
    parser.add_argument("--trigger-pct", type=float, default=25.0, help="Rainfall trigger percentile.")
    parser.add_argument("--exit-pct", type=float, default=5.0, help="Rainfall exit percentile.")
    parser.add_argument("--use-evt", action="store_true", help="Enable EVT tail modeling in the rainfall simulation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation reproducibility.")
    return parser.parse_args()


def print_summary(metrics):
    print("\nZARIP Execution Summary")
    print("------------------------")
    print(f"Total Exposure (ha): {metrics['total_exposure_ha']:,}")
    print(f"Total Sum Insured: ${metrics['total_sum_insured_usd']:,.0f}")
    print(f"Expected Loss: ${metrics['expected_loss_usd']:,.0f}")
    print(f"Loaded Premium Pool: ${metrics['loaded_premium_usd']:,.0f}")
    print(f"VaR 95%: ${metrics['VaR95_usd']:,.0f}")
    print(f"VaR 99%: ${metrics['VaR99_usd']:,.0f}")
    print(f"CVaR 95%: ${metrics['CVaR95_usd']:,.0f}")
    print(f"Contingent Liability 95%: ${metrics['contingent_liability_95_usd']:,.0f}")
    print(f"Probability of Any Payout: {metrics['prob_any_payout']:.1%}")
    print(f"Output folder: {os.path.abspath(args.output_dir)}\n")


def main():
    global args
    args = parse_args()
    if not args.nogui:
        logger.info("Launching ZARIP desktop application")
        app = ZARIPApp()
        app.mainloop()
        return

    logger.info("Running ZARIP pipeline in CLI mode")
    os.makedirs(args.output_dir, exist_ok=True)
    loader = DataLoader()
    if args.csv:
        loader.load_csv(args.csv)
        logger.info(f"Loaded historical data from {args.csv}")
    else:
        loader.load_default()
        logger.info("Loaded default synthetic dataset")

    metrics, regional_df, sens_df, cleaned_df, sim_rain, sim_payouts, rainfall_model, ins_engine = run_zarip_pipeline(
        output_dir=args.output_dir,
        n_simulations=args.iterations,
        sum_insured_usd_per_ha=args.sum_insured,
        premium_loading=args.premium_loading,
        trigger_pct=args.trigger_pct,
        exit_pct=args.exit_pct,
        use_evt=args.use_evt,
        raw_df=loader.df,
        seed=args.seed,
    )
    print_summary(metrics)


if __name__ == "__main__":
    main()
