import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import DEFAULT_REGIONAL_EXPOSURES, PALETTE, REGIONS
from .data import DataLoader
from .pipeline import run_zarip_pipeline


class ZARIPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ZARIP Desktop Tool — Zimbabwe Agricultural Risk Insurance Platform")
        self.geometry("1220x820")
        self.minsize(1080, 720)
        self.configure(bg=PALETTE["bg_dark"])

        self.loader = DataLoader()
        self.results = {}
        self._sim_thread = None

        self._build_style()
        self._build_layout()
        self._load_default_data()

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=PALETTE["bg_dark"])
        style.configure("TLabel", background=PALETTE["bg_dark"], foreground=PALETTE["text"])
        style.configure("TButton", background=PALETTE["highlight"], foreground=PALETTE["text"], borderwidth=0, padding=(8, 6))
        style.map("TButton", background=[("active", PALETTE["accent"]), ("pressed", PALETTE["accent2"])])
        style.configure("Accent.TButton", background=PALETTE["accent"], foreground=PALETTE["bg_dark"], font=(None, 10, "bold"))
        style.configure("TEntry", fieldbackground=PALETTE["bg_panel"], foreground=PALETTE["text"], insertcolor=PALETTE["text"], bordercolor=PALETTE["border"])
        style.configure("Horizontal.TProgressbar", troughcolor=PALETTE["bg_panel"], background=PALETTE["accent"], borderwidth=0)
        style.configure("Treeview", background=PALETTE["bg_panel"], fieldbackground=PALETTE["bg_panel"], foreground=PALETTE["text"], rowheight=24)
        style.configure("Treeview.Heading", background=PALETTE["highlight"], foreground=PALETTE["accent"], relief="flat")

    def _build_layout(self):
        header = tk.Frame(self, bg=PALETTE["bg_mid"], height=56)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="ZARIP", bg=PALETTE["bg_mid"], fg=PALETTE["accent"], font=(None, 18, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(header, text="Zimbabwe Agricultural Risk Insurance Platform", bg=PALETTE["bg_mid"], fg=PALETTE["text"], font=(None, 11)).pack(side="left", pady=10)
        tk.Label(header, text="IPEC Risk Toolkit", bg=PALETTE["bg_mid"], fg=PALETTE["text_dim"], font=(None, 9)).pack(side="right", padx=16, pady=10)

        body = tk.Frame(self, bg=PALETTE["bg_dark"])
        body.pack(fill="both", expand=True, padx=12, pady=8)

        left_panel = tk.Frame(body, bg=PALETTE["bg_panel"], width=320, highlightbackground=PALETTE["border"], highlightthickness=1)
        left_panel.pack(side="left", fill="y", padx=(0, 8))
        left_panel.pack_propagate(False)
        self._build_controls(left_panel)

        right_panel = tk.Frame(body, bg=PALETTE["bg_dark"])
        right_panel.pack(side="left", fill="both", expand=True)
        self._build_notebook(right_panel)

        status_bar = tk.Frame(self, bg=PALETTE["bg_mid"], height=30)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready. Load data or run default simulation.")
        self._progress_var = tk.DoubleVar(value=0.0)
        tk.Label(status_bar, textvariable=self._status_var, bg=PALETTE["bg_mid"], fg=PALETTE["text_dim"], font=(None, 9)).pack(side="left", padx=10)
        ttk.Progressbar(status_bar, variable=self._progress_var, maximum=100, style="Horizontal.TProgressbar", length=220).pack(side="right", padx=12, pady=4)

    def _build_controls(self, parent):
        def section(title):
            tk.Label(parent, text=title.upper(), bg=PALETTE["bg_panel"], fg=PALETTE["accent"], font=(None, 8, "bold")).pack(anchor="w", padx=12, pady=(14, 4))
            ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=(0, 8))

        def labeled_entry(label, default):
            row = tk.Frame(parent, bg=PALETTE["bg_panel"])
            row.pack(fill="x", padx=12, pady=4)
            tk.Label(row, text=label, bg=PALETTE["bg_panel"], fg=PALETTE["text_dim"], font=(None, 9)).pack(side="left")
            var = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=var, width=10).pack(side="right")
            return var

        section("Data")
        row_data = tk.Frame(parent, bg=PALETTE["bg_panel"])
        row_data.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(row_data, text="Load CSV", command=self._load_csv).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(row_data, text="Default Data", command=self._load_default_data).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self._data_label = tk.Label(parent, text="No dataset loaded", bg=PALETTE["bg_panel"], fg=PALETTE["text_dim"], font=(None, 8))
        self._data_label.pack(fill="x", padx=12)

        section("Insurance Parameters")
        self._v_sum_insured = labeled_entry("Sum Insured ($/ha)", 400)
        self._v_premium_loading = labeled_entry("Premium Loading", 1.25)
        self._v_trigger_pct = labeled_entry("Trigger Percentile", 25)
        self._v_exit_pct = labeled_entry("Exit Percentile", 5)

        section("Yield Response")
        self._v_ymax = labeled_entry("Y_max (t/ha)", 2.42)
        self._v_k = labeled_entry("k slope", 0.0183)
        self._v_r0 = labeled_entry("R0", 467)

        section("Simulation")
        self._v_iterations = labeled_entry("Iterations", 100000)
        self._v_use_evt = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="Enable EVT tail modeling", variable=self._v_use_evt, bg=PALETTE["bg_panel"], fg=PALETTE["text"], selectcolor=PALETTE["bg_panel"], activebackground=PALETTE["bg_panel"], activeforeground=PALETTE["accent"], highlightthickness=0).pack(anchor="w", padx=12, pady=4)

        ttk.Button(parent, text="Run Simulation", style="Accent.TButton", command=self._run_simulation).pack(fill="x", padx=12, pady=(12, 4))
        ttk.Button(parent, text="Export Results CSV", command=self._export_results).pack(fill="x", padx=12)

        section("Regional Exposure")
        tree = ttk.Treeview(parent, columns=("Region", "Hectares"), show="headings", height=5)
        tree.heading("Region", text="Region")
        tree.heading("Hectares", text="Hectares")
        tree.column("Region", width=170, anchor="w")
        tree.column("Hectares", width=110, anchor="e")
        for code, info in REGIONS.items():
            tree.insert("", "end", values=(info["name"], f"{int(info['ha']):,}"))
        tree.pack(fill="x", padx=10, pady=(0, 12))

    def _build_notebook(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        self._chart_tab = tk.Frame(notebook, bg=PALETTE["bg_dark"])
        self._summary_tab = tk.Frame(notebook, bg=PALETTE["bg_dark"])
        self._premium_tab = tk.Frame(notebook, bg=PALETTE["bg_dark"])

        notebook.add(self._chart_tab, text="Loss Distribution")
        notebook.add(self._summary_tab, text="Risk Metrics")
        notebook.add(self._premium_tab, text="Premium Table")

        self._build_chart_tab()
        self._build_summary_tab()
        self._build_premium_tab()

    def _build_chart_tab(self):
        self._fig = Figure(figsize=(8, 5), facecolor=PALETTE["bg_dark"])
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(PALETTE["bg_mid"])
        self._ax.tick_params(colors=PALETTE["text_dim"])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        self._ax.set_title("Run a simulation to show portfolio loss distribution", color=PALETTE["text"], fontsize=11)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._chart_tab)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def _build_summary_tab(self):
        self._summary_text = tk.Text(self._summary_tab, bg=PALETTE["bg_panel"], fg=PALETTE["text"], relief="flat", wrap="word", padx=14, pady=12, state="disabled", cursor="arrow")
        self._summary_text.pack(fill="both", expand=True, padx=10, pady=10)
        self._summary_text.tag_configure("title", foreground=PALETTE["accent"], font=(None, 11, "bold"))
        self._summary_text.tag_configure("section", foreground=PALETTE["accent2"], font=(None, 10, "bold"))
        self._summary_text.tag_configure("value", foreground=PALETTE["text"], font=(None, 10))
        self._summary_text.tag_configure("dim", foreground=PALETTE["text_dim"], font=(None, 9))
        self._summary_text.tag_configure("warn", foreground=PALETTE["danger"], font=(None, 10, "bold"))

    def _build_premium_tab(self):
        cols = ("Region", "Exposure (ha)", "Pure Premium (%)", "Loaded Premium (%)", "Expected Loss ($M)", "VaR95 ($M)")
        self._premium_tree = ttk.Treeview(self._premium_tab, columns=cols, show="headings", height=14)
        widths = [160, 110, 120, 120, 120, 120]
        for col, width in zip(cols, widths):
            self._premium_tree.heading(col, text=col)
            self._premium_tree.column(col, width=width, anchor="center")
        scrollbar = ttk.Scrollbar(self._premium_tab, orient="vertical", command=self._premium_tree.yview)
        self._premium_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self._premium_tree.pack(fill="both", expand=True, padx=8, pady=8)

    def _load_csv(self):
        path = filedialog.askopenfilename(title="Select historical data CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.loader.load_csv(path)
            self._data_label.config(text=f"Loaded {os.path.basename(path)} ({len(self.loader.df)} rows)", fg=PALETTE["accent"])
            self._status("CSV loaded successfully.")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _load_default_data(self):
        self.loader.load_default()
        self._data_label.config(text=f"Default sample data loaded ({len(self.loader.df)} rows)", fg=PALETTE["accent"])
        self._status("Default dataset loaded.")

    def _run_simulation(self):
        if self.loader.df is None:
            messagebox.showwarning("No Data", "Load data or use default data before running the simulation.")
            return
        try:
            n_iter = int(float(self._v_iterations.get()))
            n_iter = max(10000, min(500000, n_iter))
            sum_insured = float(self._v_sum_insured.get())
            premium_loading = float(self._v_premium_loading.get())
            trigger_pct = float(self._v_trigger_pct.get())
            exit_pct = float(self._v_exit_pct.get())
            use_evt = self._v_use_evt.get()
        except ValueError as exc:
            messagebox.showerror("Input Error", f"Invalid parameter: {exc}")
            return

        self._progress_var.set(0)
        self._status(f"Running {n_iter:,} iterations…")
        self._sim_thread = threading.Thread(target=self._execute_simulation, args=(n_iter, sum_insured, premium_loading, trigger_pct, exit_pct, use_evt), daemon=True)
        self._sim_thread.start()

    def _execute_simulation(self, n_iter, sum_insured, premium_loading, trigger_pct, exit_pct, use_evt):
        try:
            metrics, regional_df, sens_df, cleaned_df, sim_rain, sim_payouts, rainfall_model, ins_engine = run_zarip_pipeline(
                output_dir="output",
                n_simulations=n_iter,
                sum_insured_usd_per_ha=sum_insured,
                premium_loading=premium_loading,
                trigger_pct=trigger_pct,
                exit_pct=exit_pct,
                use_evt=use_evt,
                raw_df=self.loader.df,
                seed=42,
            )
            self.results = {
                "metrics": metrics,
                "regional_df": regional_df,
                "sens_df": sens_df,
                "cleaned_df": cleaned_df,
                "sim_rain": sim_rain,
                "sim_payouts": sim_payouts,
                "rainfall_model": rainfall_model,
                "ins_engine": ins_engine,
                "params": {
                    "n_iter": n_iter,
                    "sum_insured": sum_insured,
                    "premium_loading": premium_loading,
                    "trigger_pct": trigger_pct,
                    "exit_pct": exit_pct,
                    "use_evt": use_evt,
                },
            }
            self.after(0, self._on_simulation_complete)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Simulation Error", str(exc)))
            self.after(0, lambda: self._status("Simulation failed."))

    def _on_simulation_complete(self):
        self._progress_var.set(100)
        self._status("Simulation complete.")
        self._update_chart()
        self._update_summary()
        self._update_premium_table()

    def _update_chart(self):
        sim_payouts = self.results["sim_payouts"]
        portfolio_losses = np.zeros(len(next(iter(sim_payouts.values()))))
        for reg, payouts in sim_payouts.items():
            portfolio_losses += payouts * DEFAULT_REGIONAL_EXPOSURES[reg]

        losses_m = portfolio_losses / 1e6
        metrics = self.results["metrics"]

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(PALETTE["bg_mid"])
        ax.tick_params(colors=PALETTE["text_dim"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])

        ax.hist(losses_m, bins=80, color=PALETTE["accent"], alpha=0.75, edgecolor="none")
        var95_m = metrics['VaR95_usd'] / 1e6
        var99_m = metrics['VaR99_usd'] / 1e6
        cvar95_m = metrics['CVaR95_usd'] / 1e6
        el_m = metrics['expected_loss_usd'] / 1e6

        ax.axvline(el_m, color="#60A5FA", linestyle="--", linewidth=1.6, label=f"E[L] ${el_m:.2f}M")
        ax.axvline(var95_m, color=PALETTE["accent2"], linestyle="-.", linewidth=2.0, label=f"VaR 95% ${var95_m:.2f}M")
        ax.axvline(var99_m, color=PALETTE["danger"], linestyle="-.", linewidth=2.0, label=f"VaR 99% ${var99_m:.2f}M")
        ax.axvline(cvar95_m, color="#E879F9", linestyle=":", linewidth=1.5, label=f"CVaR 95% ${cvar95_m:.2f}M")

        ax.set_xlabel("Portfolio Loss (USD Millions)", color=PALETTE["text_dim"])
        ax.set_ylabel("Frequency", color=PALETTE["text_dim"])
        ax.set_title("Simulated Portfolio Loss Distribution", color=PALETTE["text"], fontsize=11)
        ax.legend(facecolor=PALETTE["bg_panel"], edgecolor=PALETTE["border"], fontsize=9)
        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()

    def _update_summary(self):
        metrics = self.results["metrics"]
        params = self.results["params"]

        self._summary_text.config(state="normal")
        self._summary_text.delete("1.0", "end")

        def put(text, tag="value"):
            self._summary_text.insert("end", text + "\n", tag)

        put("ZARIP Risk Metrics Summary", "title")
        put("")
        put("Simulation Parameters", "section")
        put(f"  Iterations: {params['n_iter']:,}")
        put(f"  Sum Insured: ${params['sum_insured']:,.2f} per ha")
        put(f"  Premium Loading: {params['premium_loading']:.2f}")
        put(f"  Trigger: {params['trigger_pct']}th percentile")
        put(f"  Exit: {params['exit_pct']}th percentile")
        put(f"  EVT Enabled: {params['use_evt']}")
        put("")
        put("Portfolio Loss Metrics", "section")
        put(f"  Expected Loss: ${metrics['expected_loss_usd']:,.0f}")
        put(f"  VaR 95%: ${metrics['VaR95_usd']:,.0f}")
        put(f"  CVaR 95%: ${metrics['CVaR95_usd']:,.0f}")
        put(f"  Tail Ratio: {metrics['tail_ratio']:.2f}")
        put(f"  Loaded Premium Pool: ${metrics['loaded_premium_usd']:,.0f}")
        put(f"  Contingent Fiscal Liability (95%): ${metrics['contingent_liability_95_usd']:,.0f}")
        put(f"  Probability of Any Payout: {metrics['prob_any_payout']:.1%}")
        put(f"  Std Dev of Loss: ${metrics['std_loss_usd']:,.0f}")
        put(f"  Max Simulated Loss: ${metrics['max_loss_usd']:,.0f}")
        self._summary_text.config(state="disabled")

    def _update_premium_table(self):
        for row in self._premium_tree.get_children():
            self._premium_tree.delete(row)
        regional_df = self.results["regional_df"]
        for _, row in regional_df.iterrows():
            self._premium_tree.insert("", "end", values=(
                row['region'].replace("_", " "),
                f"{int(row['exposure_ha']):,}",
                f"{row['pure_premium_rate_pct']:.2f}%",
                f"{row['loaded_premium_rate_pct']:.2f}%",
                f"{row['expected_loss_usd'] / 1e6:.2f}",
                f"{row['VaR95_usd'] / 1e6:.2f}",
            ))

    def _export_results(self):
        if not self.results:
            messagebox.showinfo("No Results", "Run a simulation first before exporting.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Export Results")
        if not path:
            return
        metrics = self.results["metrics"]
        regional_df = self.results["regional_df"]
        params = self.results["params"]
        with open(path, "w", newline="") as f:
            writer = __import__('csv').writer(f)
            writer.writerow(["ZARIP Results Export", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")])
            writer.writerow([])
            writer.writerow(["-- Parameters --"])
            for k, v in params.items():
                writer.writerow([k, v])
            writer.writerow([])
            writer.writerow(["-- Portfolio Metrics --"])
            for key in ["expected_loss_usd", "VaR95_usd", "VaR99_usd", "CVaR95_usd", "CVaR99_usd", "tail_ratio", "loaded_premium_usd", "contingent_liability_95_usd", "prob_any_payout", "max_loss_usd", "std_loss_usd"]:
                writer.writerow([key, metrics[key]])
            writer.writerow([])
            writer.writerow(["-- Regional Metrics --"])
            writer.writerow(regional_df.columns.tolist())
            for _, row in regional_df.iterrows():
                writer.writerow(row.tolist())
        messagebox.showinfo("Export Complete", f"Results saved to:\n{path}")
        self._status(f"Exported results to {os.path.basename(path)}")

    def _status(self, message: str):
        self._status_var.set(message)
        self.update_idletasks()
