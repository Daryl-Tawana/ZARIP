#!/usr/bin/env python
"""
Launch script for the ZARIP Flask application.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from app import app


if __name__ == "__main__":
    print(
        "\n"
        "ZARIP - Zimbabwe Agricultural Risk Insurance Platform v2.0\n"
        "Developed for the Insurance and Pensions Commission of Zimbabwe (IPEC)\n\n"
        "Features:\n"
        "- Monte Carlo VaR/CVaR climate stress testing\n"
        "- Policy scorecard for fiscal savings, SCR, and rollout planning\n"
        "- Smart subsidy allocation and basis risk disclosure\n"
        "- ZARIP AI assistant for result interpretation\n\n"
        "Open: http://127.0.0.1:5000\n"
        "Press CTRL+C to stop the server.\n"
    )

    app.run(debug=True, host="127.0.0.1", port=5000)
