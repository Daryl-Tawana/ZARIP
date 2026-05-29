#!/usr/bin/env python
"""
Launch script for ZARIP Flask application
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from app import app

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║     🚀 ZARIP - Zimbabwe Agricultural Risk Insurance Platform v2.0            ║
    ║                                                                              ║
    ║     Developed for: Insurance and Pensions Commission of Zimbabwe (IPEC)      ║
    ║                                                                              ║
    ║     🤝 Leaving no one and no place behind                                    ║
    ║                                                                              ║
    ║     Features:                                                                ║
    ║     • Monte Carlo Simulation (VaR/CVaR analysis)                             ║
    ║     • Extreme Value Theory for tail risk                                     ║
    ║     • Smart subsidy allocation (25-70% tiered)                               ║
    ║     • AI-powered assistant for interpretation                                ║
    ║     • Financial literacy education                                           ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    🌐 Starting ZARIP server...
    
    📱 Open your browser and navigate to: http://127.0.0.1:5000
    
    💡 Tip: Run a simulation first, then ask the AI Assistant questions!
    
    🔴 Press CTRL+C to stop the server.
    """)
    
    app.run(debug=True, host='127.0.0.1', port=5000)