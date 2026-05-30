"""
ZARIP (Zimbabwe Agricultural Risk Insurance Platform) - Flask Web Application
=======================================================================
A modern, AI-powered tool for climate risk assessment and agricultural insurance pricing.

Developed by: The Actuarial Team
For: Insurance and Pensions Commission of Zimbabwe (IPEC)
Date: May 2026
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback
import json
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)

# ============================================================
# DEFAULT PARAMETERS (from the policy paper)
# ============================================================
DEFAULT_PARAMS = {
    'sum_insured': 400.0,
    'premium_loading': 1.25,
    'trigger_percentile': 25,
    'exit_percentile': 5,
    'n_iterations': 10000,
    'spatial_decay': 200.0,
    'yield_max': 2.42,
    'yield_k': 0.0183,
    'yield_r0': 467.0,
    'basis_risk_sigma': 0.15,
    'evt_xi': -0.25,
    'evt_sigma': 45.0,
    'evt_threshold_pct': 10,
    'climate_trend': 0.0
}

# ============================================================
# POLICY TARGETS (from the IPEC policy paper)
# ============================================================
POLICY_TARGETS = {
    'household_plot_ha': 0.5,
    'target_households': 1500000,
    'reactive_relief_event_usd': 200e6,
    'annual_fiscal_saving_low_usd': 42e6,
    'annual_fiscal_saving_high_usd': 65e6,
    'prefunded_treasury_cost_low_usd': 15e6,
    'prefunded_treasury_cost_high_usd': 25e6,
    'credit_unlocked_low_usd': 100e6,
    'credit_unlocked_high_usd': 200e6,
    'yield_lift_low_pct': 15,
    'yield_lift_high_pct': 25,
    'capital_standard': 'CVaR 99%',
    'basis_risk_correlation_floor': 0.60,
    'basis_risk_sigma_t_ha': 0.15,
    'sensitivity_rankings': [
        {'parameter': 'Rainfall Mean', 'low_var95_pct': -99.25, 'high_var95_pct': 68.12},
        {'parameter': 'Insurance Trigger Level', 'low_var95_pct': -14.45, 'high_var95_pct': 32.83},
        {'parameter': 'Rainfall Std Dev', 'low_var95_pct': 19.78, 'high_var95_pct': -24.82},
        {'parameter': 'Basis Risk Std Dev', 'low_var95_pct': 0.88, 'high_var95_pct': 0.88},
        {'parameter': 'EVT Tail Probability', 'low_var95_pct': 0.88, 'high_var95_pct': 0.88},
        {'parameter': 'Yield Steepness k', 'low_var95_pct': 0.88, 'high_var95_pct': 0.88},
    ],
    'implementation_phases': [
        {
            'phase': 'Foundation',
            'timeframe': '0-6 months',
            'actions': [
                'Establish the National Climate Risk Database',
                'Secure data-sharing MOUs between IPEC, MSD, ZIMSTAT, and Ministry of Agriculture',
                'Draft parametric insurance amendments to the Insurance Act'
            ]
        },
        {
            'phase': 'Pilot',
            'timeframe': '6-18 months',
            'actions': [
                'Pilot VaR-based stress testing with 3-4 insurers',
                'Run farmer education campaigns in pilot districts',
                'Refine the smart subsidy mechanism using pilot feedback'
            ]
        },
        {
            'phase': 'Roll-out',
            'timeframe': '18-36 months',
            'actions': [
                'Launch national parametric product for the 2027/28 season',
                'Integrate CVaR 99% into ZICARP agricultural SCR',
                'Establish a contingent credit or catastrophe bond backstop'
            ]
        }
    ]
}

# ============================================================
# REGIONAL DATA (from paper Table 3.2)
# ============================================================
REGIONS = {
    'Region I (Manicaland)': {
        'exposure_ha': 25000,
        'subsidy_rate': 0.25,
        'log_mu': 6.851,
        'log_sigma': 0.147,
        'description': 'High rainfall area, commercial farming zone',
        'color': '#3498db',
        'icon': '🌧️'
    },
    'Region II (Mashonaland)': {
        'exposure_ha': 225000,
        'subsidy_rate': 0.25,
        'log_mu': 6.678,
        'log_sigma': 0.162,
        'description': 'Core breadbasket, medium-high rainfall',
        'color': '#2ecc71',
        'icon': '🌾'
    },
    'Region III (Midlands)': {
        'exposure_ha': 200000,
        'subsidy_rate': 0.50,
        'log_mu': 6.545,
        'log_sigma': 0.171,
        'description': 'Transition zone, moderate rainfall variability',
        'color': '#f39c12',
        'icon': '⛰️'
    },
    'Region IV (Masvingo)': {
        'exposure_ha': 175000,
        'subsidy_rate': 0.60,
        'log_mu': 6.296,
        'log_sigma': 0.198,
        'description': 'Low rainfall, high vulnerability zone',
        'color': '#e67e22',
        'icon': '🏜️'
    },
    'Region V (Matabeleland S)': {
        'exposure_ha': 125000,
        'subsidy_rate': 0.70,
        'log_mu': 6.099,
        'log_sigma': 0.221,
        'description': 'Arid region, highest drought risk',
        'color': '#e74c3c',
        'icon': '🔥'
    }
}

REGION_NAMES = list(REGIONS.keys())
N_REGIONS = len(REGION_NAMES)

# Region coordinates for spatial correlation
REGION_COORDS = {
    'Region I (Manicaland)': (32.7, -18.2),
    'Region II (Mashonaland)': (31.0, -17.5),
    'Region III (Midlands)': (29.8, -19.4),
    'Region IV (Masvingo)': (30.8, -20.1),
    'Region V (Matabeleland S)': (29.0, -21.0)
}

# ============================================================
# AI ASSISTANT KNOWLEDGE BASE
# ============================================================
AI_KNOWLEDGE = {
    'var': {
        'title': 'What is Value-at-Risk (VaR)?',
        'content': 'VaR is a statistical measure that estimates the maximum potential loss over a specified time period at a given confidence level. For example, a 95% VaR of $175M means there is a 95% probability that losses will not exceed $175M in any given year, or a 5% chance they will exceed it.',
        'example': 'If you have a 95% VaR of $100, you can be 95% confident you won\'t lose more than $100 in the specified period.'
    },
    'cvar': {
        'title': 'What is Conditional Value-at-Risk (CVaR)?',
        'content': 'CVaR (also called Expected Shortfall) measures the average loss in the worst cases beyond the VaR threshold. It answers: "If things go really bad, how bad on average?" This is more conservative than VaR.',
        'example': 'If VaR is $175M but CVaR is $216M, the average loss in the worst 5% of scenarios is $216M – $41M more than the VaR threshold.'
    },
    'tail_ratio': {
        'title': 'What is the Tail Ratio?',
        'content': 'The Tail Ratio = CVaR / Expected Loss. It shows how severe extreme losses are compared to average losses. A ratio above 2.5 indicates very heavy tail risk requiring extra capital reserves.',
        'example': 'A tail ratio of 4.59 means losses in bad years are nearly 5 times the average loss – a strong signal for tail risk.'
    },
    'basis_risk': {
        'title': 'What is Basis Risk?',
        'content': 'Basis risk is the mismatch between an insurance index (like rainfall measured at a weather station) and actual farm conditions. It can cause "false negatives" (crop fails but no payout) or "false positives" (payout when crop is fine).',
        'example': 'A farmer 20km from the weather station may have drought while the station records normal rainfall – that\'s basis risk.'
    },
    'evt': {
        'title': 'What is Extreme Value Theory (EVT)?',
        'content': 'EVT is a statistical method for estimating the probability of rare, extreme events that haven\'t occurred in the historical record (like 1-in-100-year droughts). It uses the Generalised Pareto Distribution to model the tail of the distribution.',
        'example': 'With only 45 years of data, EVT can estimate the probability of a 1-in-100-year drought that hasn\'t happened yet.'
    },
    'subsidy': {
        'title': 'How does the Smart Subsidy work?',
        'content': 'The smart subsidy is tiered by region – 25% for low-risk areas, up to 70% for high-risk, low-income regions. It targets fiscal resources to the most vulnerable farmers while maintaining market discipline through farmer co-payments.',
        'example': 'A farmer in Region V pays 30% of the premium; the State pays 70%. A farmer in Region I pays 75%; the State pays 25%.'
    },
    'capital': {
        'title': 'How much capital should an insurer hold?',
        'content': 'Our analysis shows holding capital equal only to expected loss ($47M) leads to insolvency 1 year in 5. Holding VaR 95% ($175M) covers 19 out of 20 years. CVaR 95% ($216M) provides a more prudent buffer for extreme tail events.',
        'example': 'Regulators should consider CVaR 99% ($267M) for Solvency Capital Requirements.'
    },
    'financial_literacy_1': {
        'title': '📚 Insurance Basics: What is Premium?',
        'content': 'A premium is the amount you pay for insurance coverage. In our model, the pure premium covers expected losses, while the loaded premium includes insurer expenses, risk margin, and profit (25% loading).'
    },
    'financial_literacy_2': {
        'title': '📚 How Index Insurance Works',
        'content': 'Unlike traditional insurance that requires a claims adjuster to visit your farm, index insurance pays automatically when an objective measurement (like rainfall or satellite vegetation data) falls below a trigger level. Payouts are faster and cheaper to administer.'
    },
    'financial_literacy_3': {
        'title': '📚 Why Diversification Matters',
        'content': 'Spreading risk across different regions reduces overall portfolio volatility because droughts are not perfectly correlated across distant areas. This allows insurers to offer lower premiums and stay solvent.'
    },
    'leaving_no_one': {
        'title': '🤝 "Leaving no one and no place behind"',
        'content': 'This principle ensures our framework reaches the most vulnerable: smallholder farmers in Regions IV and V, where rainfall is lowest and poverty rates highest. Success is measured by whether a farmer in Masvingo receives a payout before her children go hungry.'
    },
    'optimal_premium': {
        'title': 'What is the Optimal Premium?',
        'content': 'The optimal premium balances what farmers can afford (typically 5-10% of expected income) with what insurers need to stay solvent (at least the pure premium covering expected losses). Our optimisation engine recommends premiums that maximise uptake while ensuring sustainability.',
        'example': 'In Region V, the pure premium is $50/ha, loaded premium is $63/ha. With 70% subsidy, farmers pay $19/ha (5% of income) - this is optimal.'
    },
    'fair_premium': {
        'title': 'What is a Fair Premium Rate?',
        'content': 'A fair premium is actuarially sound (covers expected losses), transparent (clearly communicated), and affordable (within 5-10% of expected farm income). Differentiated premiums by region are fairer than uniform national rates.',
        'example': 'Region I (low risk) pays 19.4% premium rate; Region V (high risk) pays 15.7% - lower rate because of lower sum insured density.'
    }
}


class AIAssistant:
    """AI-powered assistant for results interpretation and financial literacy"""
    
    @staticmethod
    def interpret_results(results):
        """Generate natural language interpretation of simulation results"""
        p = results['portfolio']
        regional = results['regional']
        policy = results.get('policy_scorecard', {})
        
        interpretation = {
            'summary': "",
            'key_findings': [],
            'risk_assessment': "",
            'recommendations': [],
            'financial_literacy_tips': []
        }
        
        # Summary
        interpretation['summary'] = f"""
        The simulation analysed {p['total_exposure_ha']:,.0f} hectares across 5 agro-ecological regions 
        with {int(DEFAULT_PARAMS['n_iterations']):,} Monte Carlo iterations. The expected annual loss is 
        ${p['expected_loss_usd']/1e6:.1f} million, with a loaded premium pool of ${p['loaded_premium_usd']/1e6:.1f} million.
        """
        
        # Key findings
        if p['tail_ratio'] > 4:
            interpretation['key_findings'].append(
                f"⚠️ **Extreme tail risk detected**: Tail ratio of {p['tail_ratio']:.2f} indicates extreme losses are {p['tail_ratio']:.1f}× average losses."
            )
        else:
            interpretation['key_findings'].append(
                f"📊 **Moderate tail risk**: Tail ratio of {p['tail_ratio']:.2f} is within acceptable range."
            )
        
        if p['cfl_95_usd'] > 0:
            interpretation['key_findings'].append(
                f"🏦 **Contingent liability**: The State may need up to ${p['cfl_95_usd']/1e6:.1f}M in a 1-in-20-year drought."
            )
        
        # Find highest and lowest risk regions
        if policy:
            interpretation['key_findings'].append(
                f"**Policy coverage**: The framework covers {policy['covered_households']:,.0f} Pfumvudza/Intwasa households "
                f"({policy['access_target_pct']:.0f}% of the policy-paper target)."
            )
            interpretation['key_findings'].append(
                f"**ZICARP capital target**: Use {policy['capital_standard']} of ${policy['zicarp_scr_target_usd']/1e6:.1f}M "
                "as the prudent agricultural drought SCR benchmark."
            )

        highest_risk = max(regional, key=lambda x: x['var_95_usd'])
        lowest_risk = min(regional, key=lambda x: x['var_95_usd'])
        interpretation['key_findings'].append(
            f"📍 **Regional variation**: {highest_risk['name_short']} has highest risk (${highest_risk['var_95_usd']/1e6:.1f}M VaR 95%), "
            f"while {lowest_risk['name_short']} has lowest risk."
        )
        
        # Risk assessment
        if p['tail_ratio'] > 4.5:
            interpretation['risk_assessment'] = "HIGH RISK: The portfolio exhibits extreme tail risk. Immediate action required to secure additional capital buffers and reinsurance coverage."
        elif p['tail_ratio'] > 3:
            interpretation['risk_assessment'] = "MODERATE RISK: Tail risk is elevated but manageable with proper capital planning and diversification."
        else:
            interpretation['risk_assessment'] = "LOW RISK: The portfolio appears well-diversified with manageable tail risk."
        
        # Recommendations
        interpretation['recommendations'].append(
            f"• **Capital adequacy**: Hold capital at least at CVaR 95% (${p['cvar_95_usd']/1e6:.1f}M) rather than expected loss."
        )
        interpretation['recommendations'].append(
            f"• **Subsidy targeting**: Focus 70% subsidies on Regions IV and V where vulnerability is highest."
        )
        interpretation['recommendations'].append(
            f"• **Reinsurance**: Secure reinsurance coverage for events beyond VaR 99% (${p['var_99_usd']/1e6:.1f}M)."
        )
        
        interpretation['recommendations'].append(
            f"Policy SCR: Calibrate the agricultural solvency capital requirement to CVaR 99% (${p['cvar_99_usd']/1e6:.1f}M)."
        )
        interpretation['recommendations'].append(
            "Basis risk: Require index-farm correlation, false negative/positive frequencies, and geographic coverage limits for every parametric product."
        )

        # Financial literacy tips based on results
        interpretation['financial_literacy_tips'].extend([
            "💡 **Insurance Basics**: Premiums are calculated based on historical loss data plus a loading for expenses and profit.",
            "💡 **Risk Pooling**: Insuring multiple regions reduces overall risk because droughts aren't perfectly correlated.",
            "💡 **Tail Risk**: The most dangerous events are rare but catastrophic – that's why we use EVT to estimate them."
        ])
        
        return interpretation
    
    @staticmethod
    def answer_question(question, results=None):
        """Answer user questions based on knowledge base and simulation results"""
        question_lower = question.lower()

        if results:
            p = results.get('portfolio', {})
            policy = results.get('policy_scorecard', {})
            basis = results.get('basis_risk', {})
            regional = results.get('regional', [])

            if any(word in question_lower for word in ['summary', 'results', 'interpret', 'explain my', 'dashboard']):
                return {
                    'title': 'Current ZARIP Results Summary',
                    'content': (
                        f"The model covers {p.get('covered_households', 0):,.0f} households over "
                        f"{p.get('total_exposure_ha', 0):,.0f} ha. Expected annual claims are "
                        f"${p.get('expected_loss_usd', 0)/1e6:.2f}M, while VaR 95% is "
                        f"${p.get('var_95_usd', 0)/1e6:.2f}M and CVaR 99% is "
                        f"${p.get('cvar_99_usd', 0)/1e6:.2f}M. The policy reading is simple: budget the premium subsidy, "
                        "pre-arrange contingent finance for tail years, and use CVaR 99% as the ZICARP drought SCR benchmark."
                    )
                }

            if any(word in question_lower for word in ['scr', 'zicarp', 'capital requirement', 'capital target', 'solvency capital']):
                return {
                    'title': 'ZICARP Capital Recommendation',
                    'content': (
                        f"For supervisory capital, the policy paper recommends CVaR 99%. On this run, that is "
                        f"${p.get('cvar_99_usd', 0)/1e6:.2f}M. VaR 99% (${p.get('var_99_usd', 0)/1e6:.2f}M) is a useful "
                        "reinsurance attachment point, but CVaR is more prudent because it measures the average loss after the threshold is breached."
                    )
                }

            if any(word in question_lower for word in ['fiscal', 'treasury', 'contingent', 'liability', 'budget']):
                return {
                    'title': 'Fiscal Planning View',
                    'content': (
                        f"The annual State premium support is ${p.get('state_subsidy_usd', 0)/1e6:.2f}M. "
                        f"The 1-in-20 contingent fiscal liability is ${p.get('cfl_95_usd', 0)/1e6:.2f}M, and the "
                        f"1-in-100 liability is ${p.get('cfl_99_usd', 0)/1e6:.2f}M. The expected reform benefit remains "
                        f"${policy.get('annual_fiscal_saving_low_usd', 42e6)/1e6:.0f}-${policy.get('annual_fiscal_saving_high_usd', 65e6)/1e6:.0f}M "
                        "in annual fiscal savings compared with reactive disaster relief."
                    )
                }

            if any(word in question_lower for word in ['region', 'regional', 'highest risk', 'most risky', 'masvingo', 'matabeleland']):
                highest = max(regional, key=lambda r: r.get('var_95_usd', 0)) if regional else None
                lowest = min(regional, key=lambda r: r.get('var_95_usd', 0)) if regional else None
                rows = "\n".join(
                    f"- {r['name_short']}: loaded premium {r['loaded_premium_pct']}%, subsidy {r['subsidy_pct']:.0f}%, VaR95 ${r['var_95_usd']/1e6:.2f}M"
                    for r in regional
                )
                return {
                    'title': 'Regional Risk Breakdown',
                    'content': (
                        f"Highest VaR 95%: {highest['name_short']} (${highest['var_95_usd']/1e6:.2f}M). "
                        f"Lowest VaR 95%: {lowest['name_short']} (${lowest['var_95_usd']/1e6:.2f}M).\n\n{rows}"
                    ) if highest and lowest else "Run a simulation first so I can compare regional results."
                }

            if any(word in question_lower for word in ['basis', 'false negative', 'false positive', 'index mismatch']):
                return {
                    'title': 'Basis Risk Disclosure for This Run',
                    'content': (
                        f"Basis risk sigma is {basis.get('sigma_t_ha', 0):.2f} t/ha, giving an approximate "
                        f"+/-{basis.get('yield_confidence_band_t_ha', 0):.2f} t/ha farm-level yield band around the index. "
                        "IPEC should require each product to disclose index-farm correlation, false negative and false positive rates, and coverage-map limitations."
                    )
                }

            if any(word in question_lower for word in ['roadmap', 'implementation', 'next step', 'phase', 'rollout', 'roll-out']):
                phase_text = "\n".join(
                    f"- {phase['phase']} ({phase['timeframe']}): {', '.join(phase['actions'])}"
                    for phase in policy.get('implementation_phases', [])
                )
                return {
                    'title': 'Implementation Roadmap',
                    'content': phase_text or 'The roadmap will appear after a simulation is run.'
                }

            if any(word in question_lower for word in ['sensitivity', 'recalibration', 'trigger', 'rainfall mean']):
                rows = "\n".join(
                    f"- {row['parameter']}: -20% impact {row['low_var95_pct']}%, +20% impact {row['high_var95_pct']}%"
                    for row in policy.get('sensitivity_rankings', [])
                )
                return {
                    'title': 'Sensitivity Priorities',
                    'content': rows + "\n\nOperationally, monitor rainfall mean and trigger calibration most closely, and recalibrate triggers at least every five years."
                }
        
        # Match questions to knowledge base
        if any(word in question_lower for word in ['cvar', 'conditional var', 'expected shortfall']):
            return AI_KNOWLEDGE['cvar']
        elif any(word in question_lower for word in ['var', 'value at risk', 'value-at-risk']):
            return AI_KNOWLEDGE['var']
        elif any(word in question_lower for word in ['tail ratio', 'tail']):
            return AI_KNOWLEDGE['tail_ratio']
        elif any(word in question_lower for word in ['basis risk', 'basis']):
            return AI_KNOWLEDGE['basis_risk']
        elif any(word in question_lower for word in ['evt', 'extreme value', 'gpd']):
            return AI_KNOWLEDGE['evt']
        elif any(word in question_lower for word in ['subsidy', 'smart subsidy']):
            return AI_KNOWLEDGE['subsidy']
        elif any(word in question_lower for word in ['capital', 'solvency', 'hold']):
            return AI_KNOWLEDGE['capital']
        elif any(word in question_lower for word in ['premium', 'insurance cost']):
            return AI_KNOWLEDGE['financial_literacy_1']
        elif any(word in question_lower for word in ['index insurance', 'parametric']):
            return AI_KNOWLEDGE['financial_literacy_2']
        elif any(word in question_lower for word in ['diversification', 'diversify']):
            return AI_KNOWLEDGE['financial_literacy_3']
        elif any(word in question_lower for word in ['leaving no one', 'leave no one', 'behind']):
            return AI_KNOWLEDGE['leaving_no_one']
        elif any(word in question_lower for word in ['optimal premium', 'best premium', 'recommended premium']):
            return AI_KNOWLEDGE['optimal_premium']
        elif any(word in question_lower for word in ['fair premium', 'fair rate']):
            return AI_KNOWLEDGE['fair_premium']
        elif any(word in question_lower for word in ['affordable', 'too expensive', 'reduce premium']):
            return {
                'title': '💰 Making Insurance More Affordable',
                'content': 'There are several ways to make insurance more affordable:\n\n1. **Increase subsidies** for high-risk regions (currently 70% max)\n2. **Bundle insurance with credit** (lower effective cost)\n3. **Group policies** for farmer cooperatives (economies of scale)\n4. **Flexible payment plans** (pay after harvest)\n5. **Work-for-premium programmes** (labour contribution instead of cash)\n\nThe smart subsidy mechanism already targets 70% to the most vulnerable regions.'
            }
        else:
            # General response with available topics
            return {
                'title': '🤖 ZARIP Assistant',
                'content': f"""I can help you understand:
                
• VaR (Value-at-Risk) and CVaR (Conditional Value-at-Risk)
• Tail Ratio and what it means for risk
• Basis Risk in index insurance
• Extreme Value Theory (EVT) for rare events
• Smart Subsidy mechanism
• Capital adequacy requirements
• Insurance basics (premiums, index insurance, diversification)

Your question: "{question}"

Try asking about any of these topics!"""
            }


# ============================================================
# PREMIUM OPTIMISATION ENGINE
# ============================================================

class PremiumOptimiser:
    """Calculates optimal premium pricing for each region balancing affordability and sustainability"""
    
    @staticmethod
    def calculate_optimal_premiums(regional_results, portfolio_metrics, params):
        """
        Calculate optimal premium pricing recommendations
        
        Returns:
            dict: Optimal premiums, affordability metrics, and recommendations
        """
        optimal_premiums = []
        
        # Assumed average income per hectare per season (USD) - based on Zimbabwe smallholder data
        # Region I (high rainfall): higher yields → higher income
        # Region V (low rainfall): lower yields → lower income
        income_per_ha = {
            'Region I': 1200,
            'Region II': 1100,
            'Region III': 850,
            'Region IV': 600,
            'Region V': 400
        }
        
        # Willingness to pay coefficient (farmers are more likely to insure if premium < 10% of income)
        # Higher in high-risk regions where farmers recognise need
        wtp_coefficient = {
            'Region I': 0.08,   # 8% of income
            'Region II': 0.08,
            'Region III': 0.10,
            'Region IV': 0.12,
            'Region V': 0.15    # Highest willingness in highest risk region
        }
        
        for region in regional_results:
            region_name = region['name_short']
            region_data = next(r for r in regional_results if r['name_short'] == region_name)
            
            # Calculate metrics
            pure_premium_per_ha = region_data['pure_premium_pct'] * params['sum_insured'] / 100
            loaded_premium_per_ha = region_data['loaded_premium_pct'] * params['sum_insured'] / 100
            subsidy_rate = region_data['subsidy_pct'] / 100
            farmer_pays_per_ha = loaded_premium_per_ha * (1 - subsidy_rate)
            
            # Expected income per hectare
            expected_income = income_per_ha.get(region_name, 700)
            
            # Affordability metrics
            premium_as_pct_of_income = (farmer_pays_per_ha / expected_income) * 100
            
            # Willingness to pay threshold
            wtp_threshold = wtp_coefficient.get(region_name, 0.10) * expected_income
            
            # Determine if premium is affordable
            is_affordable = farmer_pays_per_ha <= wtp_threshold
            
            # Calculate maximum affordable premium (at WTP threshold)
            max_affordable_premium = wtp_threshold
            
            # Calculate minimum sustainable premium (break-even for insurer)
            # This is the pure premium (no profit margin)
            min_sustainable_premium = pure_premium_per_ha
            
            # Calculate optimal premium (balance between uptake and sustainability)
            # Using a weighted average: 60% weight to affordability, 40% to sustainability
            if is_affordable:
                # Premium is already affordable, could potentially increase slightly
                optimal_premium = farmer_pays_per_ha
                recommendation = "Current premium is affordable. Consider maintaining or small increase."
            else:
                # Premium is too high - recommend reduction
                optimal_premium = min(max_affordable_premium, loaded_premium_per_ha)
                recommendation = f"Premium is {premium_as_pct_of_income:.1f}% of income. Recommend increasing subsidy to {((loaded_premium_per_ha - max_affordable_premium) / loaded_premium_per_ha * 100):.0f}%."
            
            # Calculate recommended subsidy based on optimal premium
            recommended_subsidy_rate = 1 - (optimal_premium / loaded_premium_per_ha) if loaded_premium_per_ha > 0 else subsidy_rate
            recommended_subsidy_pct = max(0, min(100, recommended_subsidy_rate * 100))
            
            # Expected uptake rate based on premium affordability
            if premium_as_pct_of_income <= 5:
                expected_uptake = 0.85  # 85% of farmers would buy
            elif premium_as_pct_of_income <= 10:
                expected_uptake = 0.65
            elif premium_as_pct_of_income <= 15:
                expected_uptake = 0.45
            elif premium_as_pct_of_income <= 20:
                expected_uptake = 0.25
            else:
                expected_uptake = 0.10
            
            optimal_premiums.append({
                'region': region_name,
                'icon': region.get('icon', '📍'),
                'pure_premium_per_ha': round(pure_premium_per_ha, 2),
                'loaded_premium_per_ha': round(loaded_premium_per_ha, 2),
                'current_subsidy_pct': subsidy_rate * 100,
                'farmer_pays_current': round(farmer_pays_per_ha, 2),
                'expected_income_per_ha': expected_income,
                'premium_as_pct_of_income': round(premium_as_pct_of_income, 1),
                'is_affordable': is_affordable,
                'max_affordable_premium': round(max_affordable_premium, 2),
                'min_sustainable_premium': round(min_sustainable_premium, 2),
                'optimal_premium_recommended': round(optimal_premium, 2),
                'recommended_subsidy_pct': round(recommended_subsidy_pct, 0),
                'recommendation': recommendation,
                'expected_uptake_pct': round(expected_uptake * 100, 0),
                'risk_level': 'High' if region_data['var_95_usd'] > 50e6 else 'Medium' if region_data['var_95_usd'] > 20e6 else 'Low'
            })
        
        # Calculate overall portfolio recommendations
        total_farmer_payment = sum(p['farmer_pays_current'] * r['exposure_ha'] for p, r in zip(optimal_premiums, regional_results))
        total_optimal_farmer_payment = sum(p['optimal_premium_recommended'] * r['exposure_ha'] for p, r in zip(optimal_premiums, regional_results))
        total_subsidy_saving = total_farmer_payment - total_optimal_farmer_payment
        
        # Calculate weighted average expected uptake
        weighted_uptake = sum(p['expected_uptake_pct'] * r['exposure_ha'] for p, r in zip(optimal_premiums, regional_results)) / sum(r['exposure_ha'] for r in regional_results)
        
        return {
            'regional': optimal_premiums,
            'summary': {
                'total_farmer_payment_current_usd': round(total_farmer_payment, 2),
                'total_farmer_payment_optimal_usd': round(total_optimal_farmer_payment, 2),
                'total_subsidy_saving_usd': round(total_subsidy_saving, 2),
                'weighted_expected_uptake_pct': round(weighted_uptake, 1),
                'recommendation_summary': PremiumOptimiser.get_overall_recommendation(optimal_premiums, weighted_uptake)
            }
        }
    
    @staticmethod
    def get_overall_recommendation(optimal_premiums, weighted_uptake):
        """Generate overall policy recommendation based on premium analysis"""
        unaffordable_regions = [p for p in optimal_premiums if not p['is_affordable']]
        
        if len(unaffordable_regions) == 0:
            return "✅ All regions have affordable premiums. Current subsidy structure is effective. Consider expanding coverage."
        elif len(unaffordable_regions) <= 2:
            return f"⚠️ {len(unaffordable_regions)} region(s) have unaffordable premiums: {', '.join([r['region'] for r in unaffordable_regions])}. Recommend increasing subsidies in these regions by 10-20 percentage points."
        else:
            return f"🔴 Multiple regions have affordability issues. Recommend comprehensive subsidy review. Expected uptake would increase from current ~{weighted_uptake:.0f}% to ~{min(85, weighted_uptake + 20):.0f}% with optimal pricing."
    
    @staticmethod
    def generate_premium_chart(optimal_premiums):
        """Generate premium comparison chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        regions = [p['region'] for p in optimal_premiums]
        current = [p['farmer_pays_current'] for p in optimal_premiums]
        optimal = [p['optimal_premium_recommended'] for p in optimal_premiums]
        sustainable = [p['min_sustainable_premium'] for p in optimal_premiums]
        
        x = np.arange(len(regions))
        width = 0.25
        
        bars1 = ax.bar(x - width, current, width, label='Current Farmer Payment', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x, optimal, width, label='Recommended Optimal Premium', color='#27ae60', alpha=0.8)
        bars3 = ax.bar(x + width, sustainable, width, label='Minimum Sustainable (Break-even)', color='#3498db', alpha=0.8)
        
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Premium (USD per hectare)', fontsize=12)
        ax.set_title('Premium Optimisation Analysis: Current vs Recommended', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'${height:.0f}', (bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        return fig

# ============================================================
# SIMULATION ENGINE
# ============================================================
def build_correlation_matrix(decay_parameter):
    """Build spatial correlation matrix using exponential decay"""
    n = N_REGIONS
    corr_matrix = np.zeros((n, n))
    coords = [REGION_COORDS[name] for name in REGION_NAMES]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[j]
                dx = (lon1 - lon2) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
                dy = (lat1 - lat2) * 111.0
                distance = np.sqrt(dx**2 + dy**2)
                corr_matrix[i, j] = np.exp(-distance / decay_parameter)
    
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(corr_matrix))
    if min_eig < 1e-12:
        corr_matrix += np.eye(n) * (1e-12 - min_eig)
    return corr_matrix


def sample_lognormal_rainfall(log_mu, log_sigma, n_samples=1):
    """Sample from lognormal distribution"""
    scale = np.exp(log_mu)
    return stats.lognorm.rvs(s=log_sigma, scale=scale, size=n_samples)


def sample_gpd(xi, sigma, n_samples=1):
    """Sample from Generalised Pareto Distribution"""
    u = np.random.uniform(0, 1, n_samples)
    if abs(xi) < 1e-10:
        return -sigma * np.log(1 - u)
    else:
        return sigma * ((1 - u)**(-xi) - 1) / xi


def run_simulation(params):
    """Run Monte Carlo simulation"""
    np.random.seed(42)
    
    n_iter = int(params['n_iterations'])
    
    # Build correlation matrix
    corr_matrix = build_correlation_matrix(params['spatial_decay'])
    L = np.linalg.cholesky(corr_matrix)
    
    portfolio_losses = np.zeros(n_iter)
    regional_payouts = {name: np.zeros(n_iter) for name in REGION_NAMES}
    
    # Calculate triggers and exits for each region
    triggers = {}
    exits = {}
    for name in REGION_NAMES:
        data = REGIONS[name]
        hist_rain = sample_lognormal_rainfall(data['log_mu'], data['log_sigma'], n_samples=1000)
        triggers[name] = np.percentile(hist_rain, params['trigger_percentile'])
        exits[name] = np.percentile(hist_rain, params['exit_percentile'])
    
    # Main Monte Carlo loop
    for iteration in range(n_iter):
        z = np.random.normal(0, 1, N_REGIONS)
        z_corr = L @ z
        u_corr = stats.norm.cdf(z_corr)
        
        rainfall = []
        for idx, name in enumerate(REGION_NAMES):
            data = REGIONS[name]
            u_val = float(u_corr[idx])
            
            use_evt = np.random.random() < (params['evt_threshold_pct'] / 100)
            
            if use_evt:
                deficit = sample_gpd(params['evt_xi'], params['evt_sigma'], 1)[0]
                hist_rain = sample_lognormal_rainfall(data['log_mu'], data['log_sigma'], n_samples=1000)
                threshold = np.percentile(hist_rain, params['evt_threshold_pct'])
                rain = max(0, threshold - deficit)
            else:
                rain = stats.lognorm.ppf(u_val, s=data['log_sigma'], scale=np.exp(data['log_mu']))
                rain = max(0, rain)
            
            rainfall.append(rain)
        
        for idx, name in enumerate(REGION_NAMES):
            data = REGIONS[name]
            rain = rainfall[idx]
            
            trigger = triggers[name]
            exit_val = exits[name]
            
            if rain <= exit_val:
                payout_ratio = 1.0
            elif rain >= trigger:
                payout_ratio = 0.0
            else:
                payout_ratio = (trigger - rain) / (trigger - exit_val)
            
            payout = payout_ratio * params['sum_insured'] * data['exposure_ha']
            regional_payouts[name][iteration] = payout
            portfolio_losses[iteration] += payout
    
    return calculate_metrics(portfolio_losses, regional_payouts, params)


def calculate_basis_risk_disclosure(params):
    """Create policy-facing basis risk disclosures for index products."""
    basis_sigma = float(params.get('basis_risk_sigma', POLICY_TARGETS['basis_risk_sigma_t_ha']))
    yield_max = max(float(params.get('yield_max', DEFAULT_PARAMS['yield_max'])), 1e-9)
    coefficient_of_yield = basis_sigma / yield_max
    confidence_band = 1.96 * basis_sigma

    return {
        'sigma_t_ha': round(basis_sigma, 3),
        'coefficient_of_yield_pct': round(coefficient_of_yield * 100, 2),
        'yield_confidence_band_t_ha': round(confidence_band, 2),
        'correlation_disclosure_floor': POLICY_TARGETS['basis_risk_correlation_floor'],
        'status': 'Disclosure required',
        'interpretation': (
            f'Farm-level yields can deviate by about +/-{confidence_band:.2f} t/ha around the index yield. '
            'Products should disclose index-farm correlation, false negative/positive frequencies, and geographic limits.'
        )
    }


def build_policy_scorecard(portfolio_metrics, regional_results):
    """Translate actuarial outputs into the policy paper's expected implementation outcomes."""
    total_exposure = portfolio_metrics['total_exposure_ha']
    covered_households = total_exposure / POLICY_TARGETS['household_plot_ha']
    access_pct = covered_households / POLICY_TARGETS['target_households'] * 100
    scr_target = portfolio_metrics['cvar_99_usd']
    reinsurance_attachment = portfolio_metrics['var_99_usd']
    annual_prefunded_cost = portfolio_metrics['state_subsidy_usd']
    reserve_surplus = max(0, portfolio_metrics['loaded_premium_usd'] - portfolio_metrics['expected_loss_usd'])

    return {
        'covered_households': round(covered_households),
        'target_households': POLICY_TARGETS['target_households'],
        'access_target_pct': round(access_pct, 1),
        'zicarp_scr_target_usd': round(scr_target, 2),
        'capital_standard': POLICY_TARGETS['capital_standard'],
        'reinsurance_attachment_usd': round(reinsurance_attachment, 2),
        'reactive_relief_event_usd': round(POLICY_TARGETS['reactive_relief_event_usd'], 2),
        'annual_fiscal_saving_low_usd': round(POLICY_TARGETS['annual_fiscal_saving_low_usd'], 2),
        'annual_fiscal_saving_high_usd': round(POLICY_TARGETS['annual_fiscal_saving_high_usd'], 2),
        'prefunded_cost_low_usd': round(POLICY_TARGETS['prefunded_treasury_cost_low_usd'], 2),
        'prefunded_cost_high_usd': round(POLICY_TARGETS['prefunded_treasury_cost_high_usd'], 2),
        'modelled_state_subsidy_usd': round(annual_prefunded_cost, 2),
        'credit_unlocked_low_usd': round(POLICY_TARGETS['credit_unlocked_low_usd'], 2),
        'credit_unlocked_high_usd': round(POLICY_TARGETS['credit_unlocked_high_usd'], 2),
        'yield_lift_low_pct': POLICY_TARGETS['yield_lift_low_pct'],
        'yield_lift_high_pct': POLICY_TARGETS['yield_lift_high_pct'],
        'premium_reserves_usd': portfolio_metrics['loaded_premium_usd'],
        'reserve_surplus_usd': round(reserve_surplus, 2),
        'smart_subsidy_state_share_pct': round(
            portfolio_metrics['state_subsidy_usd'] / portfolio_metrics['loaded_premium_usd'] * 100, 1
        ) if portfolio_metrics['loaded_premium_usd'] else 0,
        'highest_subsidy_region': max(regional_results, key=lambda r: r['subsidy_pct'])['name_short'],
        'sensitivity_rankings': POLICY_TARGETS['sensitivity_rankings'],
        'implementation_phases': POLICY_TARGETS['implementation_phases']
    }


def calculate_metrics(portfolio_losses, regional_payouts, params):
    """Calculate all risk metrics"""
    sorted_losses = np.sort(portfolio_losses)
    n = len(sorted_losses)
    
    expected_loss = float(np.mean(portfolio_losses))
    loaded_premium = expected_loss * params['premium_loading']
    
    var_95 = float(np.percentile(portfolio_losses, 95))
    var_99 = float(np.percentile(portfolio_losses, 99))
    cvar_95 = float(np.mean(portfolio_losses[portfolio_losses >= var_95]))
    cvar_99 = float(np.mean(portfolio_losses[portfolio_losses >= var_99]))
    tail_ratio = cvar_95 / expected_loss if expected_loss > 0 else 0
    
    total_exposure = sum(REGIONS[name]['exposure_ha'] for name in REGION_NAMES)
    total_sum_insured = total_exposure * params['sum_insured']
    cfl_95 = max(0, var_95 - loaded_premium)
    cfl_99 = max(0, var_99 - loaded_premium)
    fiscal_saving = (
        POLICY_TARGETS['annual_fiscal_saving_low_usd'] + POLICY_TARGETS['annual_fiscal_saving_high_usd']
    ) / 2
    
    regional_results = []
    total_state_subsidy = 0.0
    total_farmer_contribution = 0.0
    for name in REGION_NAMES:
        data = REGIONS[name]
        region_payouts = regional_payouts[name]
        region_expected = float(np.mean(region_payouts))
        region_var = float(np.percentile(region_payouts, 95))
        region_cvar = float(np.mean(region_payouts[region_payouts >= region_var]))
        region_sum_insured = data['exposure_ha'] * params['sum_insured']
        pure_rate = (region_expected / region_sum_insured) * 100 if region_sum_insured > 0 else 0
        loaded_rate = pure_rate * params['premium_loading']
        loaded_premium_usd = region_expected * params['premium_loading']
        state_subsidy_usd = loaded_premium_usd * data['subsidy_rate']
        farmer_contribution_usd = loaded_premium_usd - state_subsidy_usd
        total_state_subsidy += state_subsidy_usd
        total_farmer_contribution += farmer_contribution_usd
        
        regional_results.append({
            'name': name,
            'name_short': name.split(' (')[0],
            'exposure_ha': data['exposure_ha'],
            'description': data['description'],
            'icon': data['icon'],
            'sum_insured_usd': round(region_sum_insured, 2),
            'pure_premium_pct': round(pure_rate, 2),
            'loaded_premium_pct': round(loaded_rate, 2),
            'loaded_premium_usd': round(loaded_premium_usd, 2),
            'subsidy_pct': data['subsidy_rate'] * 100,
            'state_subsidy_usd': round(state_subsidy_usd, 2),
            'farmer_contribution_usd': round(farmer_contribution_usd, 2),
            'expected_loss_usd': round(region_expected, 2),
            'var_95_usd': round(region_var, 2),
            'cvar_95_usd': round(region_cvar, 2),
            'color': data['color']
        })

    portfolio_metrics = {
        'total_exposure_ha': total_exposure,
        'covered_households': round(total_exposure / POLICY_TARGETS['household_plot_ha']),
        'total_sum_insured_usd': round(total_sum_insured, 2),
        'expected_loss_usd': round(expected_loss, 2),
        'loaded_premium_usd': round(loaded_premium, 2),
        'var_95_usd': round(var_95, 2),
        'var_99_usd': round(var_99, 2),
        'cvar_95_usd': round(cvar_95, 2),
        'cvar_99_usd': round(cvar_99, 2),
        'tail_ratio': round(tail_ratio, 2),
        'state_subsidy_usd': round(total_state_subsidy, 2),
        'farmer_contribution_usd': round(total_farmer_contribution, 2),
        'cfl_95_usd': round(cfl_95, 2),
        'cfl_99_usd': round(cfl_99, 2),
        'fiscal_saving_usd': round(fiscal_saving, 2),
        'fiscal_saving_low_usd': round(POLICY_TARGETS['annual_fiscal_saving_low_usd'], 2),
        'fiscal_saving_high_usd': round(POLICY_TARGETS['annual_fiscal_saving_high_usd'], 2)
    }

    return {
        'portfolio': {
            **portfolio_metrics
        },
        'regional': regional_results,
        'basis_risk': calculate_basis_risk_disclosure(params),
        'policy_scorecard': build_policy_scorecard(portfolio_metrics, regional_results),
        'loss_distribution': portfolio_losses.tolist()
    }


def generate_charts(results):
    """Generate all charts as base64 encoded images"""
    charts = {}
    
    # Chart 1: Loss Distribution
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    losses_m = np.array(results['loss_distribution']) / 1e6
    ax1.hist(losses_m, bins=50, color='#2c3e50', alpha=0.7, edgecolor='white', density=True)
    
    expected_m = results['portfolio']['expected_loss_usd'] / 1e6
    var95_m = results['portfolio']['var_95_usd'] / 1e6
    cvar95_m = results['portfolio']['cvar_95_usd'] / 1e6
    
    ax1.axvline(expected_m, color='#27ae60', linestyle='--', linewidth=2, label=f'Expected Loss: ${expected_m:.1f}M')
    ax1.axvline(var95_m, color='#f39c12', linestyle='--', linewidth=2, label=f'VaR 95%: ${var95_m:.1f}M')
    ax1.axvline(cvar95_m, color='#e74c3c', linestyle='--', linewidth=2, label=f'CVaR 95%: ${cvar95_m:.1f}M')
    
    ax1.set_xlabel('Portfolio Loss (USD Millions)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Portfolio Loss Distribution (Monte Carlo Simulation)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
    buf1.seek(0)
    charts['loss_distribution'] = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)
    
    # Chart 2: Premium Rates
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    regions = [r['name_short'] for r in results['regional']]
    pure = [r['pure_premium_pct'] for r in results['regional']]
    loaded = [r['loaded_premium_pct'] for r in results['regional']]
    
    x = np.arange(len(regions))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, pure, width, label='Pure Premium', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, loaded, width, label='Loaded Premium (25% loading)', color='#2c3e50', alpha=0.8)
    
    ax2.set_xlabel('Region', fontsize=12)
    ax2.set_ylabel('Premium Rate (% of Sum Insured)', fontsize=12)
    ax2.set_title('Premium Rates by Region', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, loaded):
        ax2.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    buf2.seek(0)
    charts['premium_rates'] = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)
    
    # Chart 3: Smart Subsidy Allocation
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    state_paid = [r['state_subsidy_usd'] / 1e6 for r in results['regional']]
    farmer_paid = [r['farmer_contribution_usd'] / 1e6 for r in results['regional']]
    subsidies = [r['subsidy_pct'] for r in results['regional']]

    bars_state = ax3.bar(regions, state_paid, color='#27ae60', alpha=0.85, label='State subsidy')
    bars_farmer = ax3.bar(regions, farmer_paid, bottom=state_paid, color='#f1c40f', alpha=0.85, label='Farmer contribution')
    ax3.set_xlabel('Region', fontsize=12)
    ax3.set_ylabel('Premium Volume (USD Millions)', fontsize=12)
    ax3.set_title('Smart Subsidy Allocation by Premium Volume', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, s, total in zip(bars_farmer, subsidies, np.array(state_paid) + np.array(farmer_paid)):
        ax3.annotate(f'{s:.0f}% state', (bar.get_x() + bar.get_width()/2, total),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', dpi=150, bbox_inches='tight')
    buf3.seek(0)
    charts['subsidy_allocation'] = base64.b64encode(buf3.read()).decode('utf-8')
    plt.close(fig3)
    
    # Chart 4: Risk Heatmap
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    risk = [r['var_95_usd'] / 1e6 for r in results['regional']]
    risk_colors = plt.cm.RdYlGn_r(np.array(risk) / max(risk))
    
    bars = ax4.bar(regions, risk, color=risk_colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax4.set_xlabel('Region', fontsize=12)
    ax4.set_ylabel('VaR 95% (USD Millions)', fontsize=12)
    ax4.set_title('Risk Heatmap by Region', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, r in zip(bars, risk):
        ax4.annotate(f'${r:.1f}M', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    
    buf4 = io.BytesIO()
    plt.savefig(buf4, format='png', dpi=150, bbox_inches='tight')
    buf4.seek(0)
    charts['risk_heatmap'] = base64.b64encode(buf4.read()).decode('utf-8')
    plt.close(fig4)
    
    return charts


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', regions=REGIONS, default_params=DEFAULT_PARAMS)


@app.route('/assistant')
def assistant():
    """AI Assistant page"""
    return render_template('assistant.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run simulation and return results"""
    try:
        data = request.get_json()
        params = DEFAULT_PARAMS.copy()
        if data and 'params' in data:
            for key, value in data['params'].items():
                if key in params:
                    params[key] = float(value) if isinstance(value, (int, float)) else value
        
        results = run_simulation(params)
        charts = generate_charts(results)
        ai_interpretation = AIAssistant.interpret_results(results)
        
        return jsonify({
            'success': True,
            'results': results,
            'charts': charts,
            'ai_interpretation': ai_interpretation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """AI Assistant question endpoint"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        results = data.get('results', None)
        
        answer = AIAssistant.answer_question(question, results)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_results():
    """Export results to Excel"""
    try:
        data = request.get_json()
        results = data.get('results')
        
        if not results:
            return jsonify({'success': False, 'error': 'No results to export'}), 400
        
        portfolio_df = pd.DataFrame([results['portfolio']])
        regional_df = pd.DataFrame(results['regional'])
        policy_df = pd.DataFrame([results.get('policy_scorecard', {})])
        basis_df = pd.DataFrame([results.get('basis_risk', {})])
        sensitivity_df = pd.DataFrame(results.get('policy_scorecard', {}).get('sensitivity_rankings', []))
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            portfolio_df.to_excel(writer, sheet_name='Portfolio Metrics', index=False)
            regional_df.to_excel(writer, sheet_name='Regional Results', index=False)
            policy_df.to_excel(writer, sheet_name='Policy Scorecard', index=False)
            basis_df.to_excel(writer, sheet_name='Basis Risk Disclosure', index=False)
            sensitivity_df.to_excel(writer, sheet_name='Sensitivity Rankings', index=False)
            
            # Add interpretation sheet
            if 'ai_interpretation' in data:
                interpretation_data = data['ai_interpretation']
                summary_df = pd.DataFrame([{'Summary': interpretation_data.get('summary', '')}])
                summary_df.to_excel(writer, sheet_name='AI Analysis', index=False)
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'zarip_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/optimise-premiums', methods=['POST'])
def optimise_premiums():
    """Calculate optimal premium pricing recommendations"""
    try:
        data = request.get_json()
        results = data.get('results')
        params = data.get('params', DEFAULT_PARAMS)
        
        if not results:
            return jsonify({'success': False, 'error': 'No simulation results provided'}), 400
        
        optimiser = PremiumOptimiser()
        recommendations = optimiser.calculate_optimal_premiums(
            results['regional'], 
            results['portfolio'], 
            params
        )
        
        # Generate chart
        fig = optimiser.generate_premium_chart(recommendations['regional'])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'chart': chart_base64
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
