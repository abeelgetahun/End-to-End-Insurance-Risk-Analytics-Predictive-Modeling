"""
Statistical Analysis Support Functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Supporting class for statistical analysis and visualization
    """
    
    def __init__(self, data):
        self.data = data
        
    def calculate_effect_sizes(self, group1, group2):
        """
        Calculate Cohen's d effect size
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        
        return cohens_d
    
    def power_analysis(self, effect_size, alpha=0.05, power=0.8):
        """
        Basic power analysis calculation
        """
        # Simplified power analysis
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size per group
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def create_comparison_plots(self, test_results, save_path=None):
        """
        Create visualization plots for hypothesis test results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Province Risk Comparison
        if 'provinces' in test_results:
            province_stats = test_results['provinces']['descriptive_stats']
            province_means = province_stats['HasClaim']['mean']
            
            axes[0, 0].bar(province_means.index, province_means.values)
            axes[0, 0].set_title('Claim Frequency by Province')
            axes[0, 0].set_ylabel('Claim Frequency')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Gender Risk Comparison
        if 'gender' in test_results:
            gender_data = self.data[self.data['Gender'].isin(['Male', 'Female'])]
            gender_claims = gender_data.groupby('Gender')['HasClaim'].mean()
            
            axes[0, 1].bar(gender_claims.index, gender_claims.values, color=['lightblue', 'lightcoral'])
            axes[0, 1].set_title('Claim Frequency by Gender')
            axes[0, 1].set_ylabel('Claim Frequency')
        
        # Plot 3: Zipcode Analysis (top 10)
        if 'zipcodes' in test_results:
            zipcode_stats = test_results['zipcodes']['descriptive_stats']
            zipcode_means = zipcode_stats['HasClaim']['mean'].head(10)
            
            axes[1, 0].bar(range(len(zipcode_means)), zipcode_means.values)
            axes[1, 0].set_title('Claim Frequency by Top 10 Zip Codes')
            axes[1, 0].set_ylabel('Claim Frequency')
            axes[1, 0].set_xticks(range(len(zipcode_means)))
            axes[1, 0].set_xticklabels(zipcode_means.index, rotation=45)
        
        # Plot 4: P-values Summary
        p_values = []
        test_names = []
        
        for category, results in test_results.items():
            if isinstance(results, dict) and 'claim_frequency' in results:
                p_values.append(results['claim_frequency']['p_value'])
                test_names.append(f"{category.title()}\nClaim Freq")
        
        if p_values:
            colors = ['red' if p < 0.05 else 'green' for p in p_values]
            axes[1, 1].bar(test_names, p_values, color=colors)
            axes[1, 1].axhline(y=0.05, color='black', linestyle='--', label='α = 0.05')
            axes[1, 1].set_title('P-values Summary')
            axes[1, 1].set_ylabel('P-value')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_statistical_report(self, test_results, save_path=None):
        """
        Generate a comprehensive statistical report
        """
        report = []
        report.append("# Statistical Hypothesis Testing Report")
        report.append("=" * 50)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        significant_tests = []
        for category, results in test_results.items():
            if isinstance(results, dict) and 'claim_frequency' in results:
                if results['claim_frequency']['reject_null']:
                    significant_tests.append(category)
        
        if significant_tests:
            report.append(f"**Significant findings detected in: {', '.join(significant_tests)}**")
        else:
            report.append("**No significant risk differences detected across tested categories.**")
        
        report.append("")
        
        # Detailed Results
        for category, results in test_results.items():
            if isinstance(results, dict):
                report.append(f"## {category.title()} Analysis")
                report.append("")
                
                for test_type, test_result in results.items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        report.append(f"### {test_type.replace('_', ' ').title()}")
                        report.append(f"- **Test Type**: {test_result['test_type']}")
                        report.append(f"- **P-value**: {test_result['p_value']:.6f}")
                        report.append(f"- **Result**: {test_result['interpretation']}")
                        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text