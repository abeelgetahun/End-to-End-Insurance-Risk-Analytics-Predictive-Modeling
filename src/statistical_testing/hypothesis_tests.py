"""
Hypothesis Testing Implementation for Insurance Risk Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

class HypothesisTests:
    """
    A comprehensive class for conducting A/B hypothesis testing on insurance data
    """
    
    def __init__(self, data):
        """
        Initialize with insurance dataset
        
        Parameters:
        -----------
        data : pd.DataFrame
            Insurance dataset with required columns
        """
        self.data = data.copy()
        self.results = {}
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for hypothesis testing"""
        # Create risk metrics
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        self.data['ClaimFrequency'] = self.data['HasClaim']
        
        # Calculate claim severity (average claim amount given a claim occurred)
        claim_data = self.data[self.data['TotalClaims'] > 0]
        if not claim_data.empty:
            self.data['ClaimSeverity'] = np.where(
                self.data['TotalClaims'] > 0, 
                self.data['TotalClaims'], 
                np.nan
            )
        else:
            self.data['ClaimSeverity'] = 0
            
        # Calculate margin (profit)
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        # Clean postal codes for zip code analysis
        self.data['PostalCode'] = self.data['PostalCode'].fillna('Unknown')
        
    def test_risk_differences_provinces(self, alpha=0.05):
        """
        Test H₀: There are no risk differences across provinces
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
            
        Returns:
        --------
        dict : Test results with p-values and conclusions
        """
        results = {}
        
        # Test 1: Claim Frequency across provinces
        province_claim_freq = pd.crosstab(self.data['Province'], self.data['HasClaim'])
        chi2_freq, p_freq, dof_freq, expected_freq = chi2_contingency(province_claim_freq)
        
        results['claim_frequency'] = {
            'test_type': 'Chi-squared test',
            'chi2_statistic': chi2_freq,
            'p_value': p_freq,
            'degrees_of_freedom': dof_freq,
            'reject_null': p_freq < alpha,
            'interpretation': f"{'Reject' if p_freq < alpha else 'Fail to reject'} H₀: Risk differences exist across provinces" if p_freq < alpha else f"{'Reject' if p_freq < alpha else 'Fail to reject'} H₀: No significant risk differences across provinces"
        }
        
        # Test 2: Claim Severity across provinces (ANOVA)
        province_groups = []
        province_names = []
        for province in self.data['Province'].unique():
            if pd.notna(province):
                severity_data = self.data[
                    (self.data['Province'] == province) & 
                    (self.data['TotalClaims'] > 0)
                ]['TotalClaims']
                if len(severity_data) > 0:
                    province_groups.append(severity_data)
                    province_names.append(province)
        
        if len(province_groups) >= 2:
            f_stat, p_severity = stats.f_oneway(*province_groups)
            results['claim_severity'] = {
                'test_type': 'One-way ANOVA',
                'f_statistic': f_stat,
                'p_value': p_severity,
                'reject_null': p_severity < alpha,
                'interpretation': f"{'Reject' if p_severity < alpha else 'Fail to reject'} H₀: {'Significant' if p_severity < alpha else 'No significant'} claim severity differences across provinces"
            }
        
        # Calculate descriptive statistics by province
        province_stats = self.data.groupby('Province').agg({
            'HasClaim': ['count', 'sum', 'mean'],
            'TotalClaims': ['mean', 'std'],
            'TotalPremium': ['mean', 'std'],
            'Margin': ['mean', 'std']
        }).round(4)
        
        results['descriptive_stats'] = province_stats
        self.results['provinces'] = results
        
        return results
    
    def test_risk_differences_zipcodes(self, alpha=0.05, top_n=10):
        """
        Test H₀: There are no risk differences between zip codes
        Focus on top N zip codes by policy count for statistical power
        
        Parameters:
        -----------
        alpha : float
            Significance level
        top_n : int
            Number of top zip codes to analyze
        """
        results = {}
        
        # Get top zip codes by count
        zipcode_counts = self.data['PostalCode'].value_counts()
        top_zipcodes = zipcode_counts.head(top_n).index.tolist()
        top_zipcode_data = self.data[self.data['PostalCode'].isin(top_zipcodes)]
        
        # Test 1: Claim Frequency across top zip codes
        zipcode_claim_freq = pd.crosstab(top_zipcode_data['PostalCode'], top_zipcode_data['HasClaim'])
        chi2_freq, p_freq, dof_freq, expected_freq = chi2_contingency(zipcode_claim_freq)
        
        results['claim_frequency'] = {
            'test_type': 'Chi-squared test',
            'chi2_statistic': chi2_freq,
            'p_value': p_freq,
            'degrees_of_freedom': dof_freq,
            'reject_null': p_freq < alpha,
            'interpretation': f"{'Reject' if p_freq < alpha else 'Fail to reject'} H₀: {'Significant' if p_freq < alpha else 'No significant'} risk differences between zip codes"
        }
        
        # Test 2: Margin differences across zip codes
        zipcode_groups = []
        zipcode_names = []
        for zipcode in top_zipcodes:
            margin_data = top_zipcode_data[top_zipcode_data['PostalCode'] == zipcode]['Margin']
            if len(margin_data) > 0:
                zipcode_groups.append(margin_data)
                zipcode_names.append(zipcode)
        
        if len(zipcode_groups) >= 2:
            f_stat, p_margin = stats.f_oneway(*zipcode_groups)
            results['margin_differences'] = {
                'test_type': 'One-way ANOVA',
                'f_statistic': f_stat,
                'p_value': p_margin,
                'reject_null': p_margin < alpha,
                'interpretation': f"{'Reject' if p_margin < alpha else 'Fail to reject'} H₀: {'Significant' if p_margin < alpha else 'No significant'} margin differences between zip codes"
            }
        
        # Descriptive statistics
        zipcode_stats = top_zipcode_data.groupby('PostalCode').agg({
            'HasClaim': ['count', 'sum', 'mean'],
            'TotalClaims': ['mean', 'std'],
            'TotalPremium': ['mean', 'std'],
            'Margin': ['mean', 'std']
        }).round(4)
        
        results['descriptive_stats'] = zipcode_stats
        results['top_zipcodes_analyzed'] = top_zipcodes
        self.results['zipcodes'] = results
        
        return results
    
    def test_margin_differences_zipcodes(self, alpha=0.05, top_n=10):
        """
        Test H₀: There are no significant margin (profit) differences between zip codes
        """
        # This is partially covered in the zipcode test above
        # Extract margin-specific results
        zipcode_results = self.test_risk_differences_zipcodes(alpha, top_n)
        
        if 'margin_differences' in zipcode_results:
            return {
                'test_results': zipcode_results['margin_differences'],
                'descriptive_stats': zipcode_results['descriptive_stats']
            }
        else:
            return {'error': 'Insufficient data for margin analysis across zip codes'}
    
    def test_risk_differences_gender(self, alpha=0.05):
        """
        Test H₀: There are no significant risk differences between Women and Men
        """
        results = {}
        
        # Filter for Male and Female only
        gender_data = self.data[self.data['Gender'].isin(['Male', 'Female'])]
        
        if len(gender_data) == 0:
            return {'error': 'No gender data available'}
        
        # Test 1: Claim Frequency by Gender (Chi-squared)
        gender_claim_freq = pd.crosstab(gender_data['Gender'], gender_data['HasClaim'])
        chi2_freq, p_freq, dof_freq, expected_freq = chi2_contingency(gender_claim_freq)
        
        results['claim_frequency'] = {
            'test_type': 'Chi-squared test',
            'chi2_statistic': chi2_freq,
            'p_value': p_freq,
            'degrees_of_freedom': dof_freq,
            'reject_null': p_freq < alpha,
            'interpretation': f"{'Reject' if p_freq < alpha else 'Fail to reject'} H₀: {'Significant' if p_freq < alpha else 'No significant'} risk differences between genders"
        }
        
        # Test 2: Claim Severity by Gender (t-test)
        male_claims = gender_data[(gender_data['Gender'] == 'Male') & (gender_data['TotalClaims'] > 0)]['TotalClaims']
        female_claims = gender_data[(gender_data['Gender'] == 'Female') & (gender_data['TotalClaims'] > 0)]['TotalClaims']
        
        if len(male_claims) > 0 and len(female_claims) > 0:
            # Check for normality (Shapiro-Wilk test on samples)
            male_sample = male_claims.sample(min(5000, len(male_claims))) if len(male_claims) > 5000 else male_claims
            female_sample = female_claims.sample(min(5000, len(female_claims))) if len(female_claims) > 5000 else female_claims
            
            _, p_male_norm = stats.shapiro(male_sample)
            _, p_female_norm = stats.shapiro(female_sample)
            
            # Use appropriate test based on normality
            if p_male_norm > 0.05 and p_female_norm > 0.05:
                # Data is normal, use t-test
                t_stat, p_severity = ttest_ind(male_claims, female_claims, equal_var=False)
                test_type = "Welch's t-test"
            else:
                # Data is not normal, use Mann-Whitney U test
                u_stat, p_severity = mannwhitneyu(male_claims, female_claims, alternative='two-sided')
                test_type = "Mann-Whitney U test"
                t_stat = u_stat
            
            results['claim_severity'] = {
                'test_type': test_type,
                'test_statistic': t_stat,
                'p_value': p_severity,
                'reject_null': p_severity < alpha,
                'interpretation': f"{'Reject' if p_severity < alpha else 'Fail to reject'} H₀: {'Significant' if p_severity < alpha else 'No significant'} claim severity differences between genders"
            }
        
        # Test 3: Total Premium by Gender
        male_premium = gender_data[gender_data['Gender'] == 'Male']['TotalPremium']
        female_premium = gender_data[gender_data['Gender'] == 'Female']['TotalPremium']
        
        t_stat_prem, p_premium = ttest_ind(male_premium, female_premium, equal_var=False)
        
        results['premium_differences'] = {
            'test_type': "Welch's t-test",
            'test_statistic': t_stat_prem,
            'p_value': p_premium,
            'reject_null': p_premium < alpha,
            'interpretation': f"{'Reject' if p_premium < alpha else 'Fail to reject'} H₀: {'Significant' if p_premium < alpha else 'No significant'} premium differences between genders"
        }
        
        # Descriptive statistics
        gender_stats = gender_data.groupby('Gender').agg({
            'HasClaim': ['count', 'sum', 'mean'],
            'TotalClaims': ['mean', 'std'],
            'TotalPremium': ['mean', 'std'],
            'Margin': ['mean', 'std']
        }).round(4)
        
        results['descriptive_stats'] = gender_stats
        self.results['gender'] = results
        
        return results
    
    def run_all_tests(self, alpha=0.05):
        """
        Run all hypothesis tests and compile comprehensive results
        """
        print("Running A/B Hypothesis Testing Suite...")
        print("=" * 50)
        
        # Test 1: Provinces
        print("Testing risk differences across provinces...")
        province_results = self.test_risk_differences_provinces(alpha)
        
        # Test 2: Zip codes
        print("Testing risk differences between zip codes...")
        zipcode_results = self.test_risk_differences_zipcodes(alpha)
        
        # Test 3: Margin differences by zip codes
        print("Testing margin differences between zip codes...")
        margin_results = self.test_margin_differences_zipcodes(alpha)
        
        # Test 4: Gender
        print("Testing risk differences between genders...")
        gender_results = self.test_risk_differences_gender(alpha)
        
        # Compile summary
        summary = {
            'provinces': province_results,
            'zipcodes': zipcode_results,
            'margins_by_zipcode': margin_results,
            'gender': gender_results,
            'alpha_level': alpha
        }
        
        self.results['summary'] = summary
        print("All tests completed!")
        
        return summary
    
    def get_business_recommendations(self):
        """
        Generate business recommendations based on test results
        """
        recommendations = []
        
        if 'summary' not in self.results:
            return ["Please run tests first using run_all_tests()"]
        
        summary = self.results['summary']
        
        # Province recommendations
        if 'provinces' in summary and 'claim_frequency' in summary['provinces']:
            if summary['provinces']['claim_frequency']['reject_null']:
                recommendations.append(
                    "🏛️ PROVINCIAL RISK ADJUSTMENT: Significant risk differences detected across provinces. "
                    "Consider implementing province-specific premium adjustments. "
                    "Analyze high-risk provinces for targeted risk mitigation strategies."
                )
            else:
                recommendations.append(
                    "🏛️ PROVINCIAL PRICING: No significant risk differences across provinces. "
                    "Current uniform provincial pricing strategy can be maintained."
                )
        
        # Zipcode recommendations
        if 'zipcodes' in summary and 'claim_frequency' in summary['zipcodes']:
            if summary['zipcodes']['claim_frequency']['reject_null']:
                recommendations.append(
                    "📍 GEOGRAPHIC SEGMENTATION: Significant risk differences between zip codes detected. "
                    "Implement zip code-level risk scoring and premium adjustments. "
                    "Consider micro-geographic factors in underwriting."
                )
        
        # Gender recommendations
        if 'gender' in summary and 'claim_frequency' in summary['gender']:
            if summary['gender']['claim_frequency']['reject_null']:
                recommendations.append(
                    "👥 GENDER-BASED INSIGHTS: Significant risk differences between genders detected. "
                    "Review current gender-neutral pricing policies. "
                    "Ensure compliance with local regulations regarding gender-based pricing."
                )
            else:
                recommendations.append(
                    "👥 GENDER EQUALITY: No significant risk differences between genders. "
                    "Current gender-neutral approach is statistically supported."
                )
        
        return recommendations