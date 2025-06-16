import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression
from sklearn.decomposition import PCA
import logging

class FeatureEngineer:
    """
    Advanced feature engineering for insurance risk modeling.
    
    Creates domain-specific features and performs feature selection
    to improve model performance and interpretability.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.feature_selectors = {}
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def create_risk_features(self, df):
        """
        Create comprehensive risk-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional risk features
        """
        df_features = df.copy()
        
        # Vehicle-based risk features
        df_features = self._create_vehicle_risk_features(df_features)
        
        # Geographic risk features
        df_features = self._create_geographic_risk_features(df_features)
        
        # Customer demographic risk features
        df_features = self._create_demographic_risk_features(df_features)
        
        # Policy-based risk features
        df_features = self._create_policy_risk_features(df_features)
        
        return df_features
    
    def _create_vehicle_risk_features(self, df):
        """Create vehicle-specific risk indicators."""
        df_risk = df.copy()
        
        # Vehicle age categories
        if 'VehicleAge' in df_risk.columns:
            df_risk['VehicleAgeCategory'] = pd.cut(
                df_risk['VehicleAge'], 
                bins=[-1, 2, 5, 10, 20, 100], 
                labels=['New', 'Recent', 'Moderate', 'Old', 'Very_Old']
            )
        
        # Engine power categories
        if 'Kilowatts' in df_risk.columns:
            df_risk['PowerCategory'] = pd.cut(
                df_risk['Kilowatts'], 
                bins=[0, 80, 120, 180, 1000], 
                labels=['Low_Power', 'Medium_Power', 'High_Power', 'Very_High_Power']
            )
        
        # Vehicle value categories
        if 'CustomValueEstimate' in df_risk.columns:
            value_quantiles = df_risk['CustomValueEstimate'].quantile([0.25, 0.5, 0.75])
            df_risk['ValueCategory'] = pd.cut(
                df_risk['CustomValueEstimate'],
                bins=[0, value_quantiles[0.25], value_quantiles[0.5], 
                      value_quantiles[0.75], df_risk['CustomValueEstimate'].max()],
                labels=['Budget', 'Economy', 'Premium', 'Luxury']
            )
        
        # Safety features score
        safety_features = ['AlarmImmobiliser', 'TrackingDevice']
        for feature in safety_features:
            if feature in df_risk.columns:
                df_risk[f'{feature}_Score'] = df_risk[feature].map({'Yes': 1, 'No': 0}).fillna(0)
        
        if all(f'{feature}_Score' in df_risk.columns for feature in safety_features):
            df_risk['SafetyScore'] = df_risk[[f'{feature}_Score' for feature in safety_features]].sum(axis=1)
        
        return df_risk
    
    def _create_geographic_risk_features(self, df):
        """Create geography-based risk indicators."""
        df_geo = df.copy()
        
        # Province risk encoding (based on historical data patterns)
        province_risk_map = {
            'Gauteng': 'High_Risk',
            'Western Cape': 'Medium_Risk', 
            'KwaZulu-Natal': 'Medium_Risk',
            'Eastern Cape': 'Low_Risk',
            'Free State': 'Low_Risk',
            'Limpopo': 'Low_Risk',
            'North West': 'Low_Risk',
            'Mpumalanga': 'Medium_Risk',
            'Northern Cape': 'Low_Risk'
        }
        
        if 'Province' in df_geo.columns:
            df_geo['ProvinceRiskLevel'] = df_geo['Province'].map(province_risk_map).fillna('Unknown')
        
        # Urban vs Rural classification (simplified based on postal codes)
        if 'PostalCode' in df_geo.columns:
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            urban_postal_prefixes = ['0001', '0002', '0003', '0004', '0005']  # Example major city codes
            df_geo['IsUrban'] = df_geo['PostalCode'].astype(str).str[:4].isin(urban_postal_prefixes).astype(int)
        
        return df_geo
    
    def _create_demographic_risk_features(self, df):
        """Create customer demographic risk indicators."""
        df_demo = df.copy()
        
        # Age groups (if age data available)
        # Note: This example assumes age isn't directly available, 
        # but you might derive it from other features
        
        # Gender risk encoding
        if 'Gender' in df_demo.columns:
            df_demo['GenderRisk'] = df_demo['Gender'].map({
                'Male': 1, 'Female': 0
            }).fillna(0.5)  # Neutral for unknown
        
        # Marital status risk
        if 'MaritalStatus' in df_demo.columns:
            marital_risk_map = {
                'Married': 0,      # Lower risk
                'Single': 1,       # Higher risk
                'Divorced': 0.5,   # Medium risk
                'Widowed': 0.3     # Lower-medium risk
            }
            df_demo['MaritalRisk'] = df_demo['MaritalStatus'].map(marital_risk_map).fillna(0.5)
        
        return df_demo
    
    def _create_policy_risk_features(self, df):
        """Create policy-specific risk indicators."""
        df_policy = df.copy()
        
        # Premium to value ratio
        if 'TotalPremium' in df_policy.columns and 'CustomValueEstimate' in df_policy.columns:
            df_policy['PremiumToValueRatio'] = df_policy['TotalPremium'] / (df_policy['CustomValueEstimate'] + 1)
        
        # Sum insured to value ratio
        if 'SumInsured' in df_policy.columns and 'CustomValueEstimate' in df_policy.columns:
            df_policy['SumInsuredToValueRatio'] = df_policy['SumInsured'] / (df_policy['CustomValueEstimate'] + 1)
        
        # Excess level categories
        if 'ExcessSelected' in df_policy.columns:
            excess_quantiles = df_policy['ExcessSelected'].quantile([0.33, 0.67])
            df_policy['ExcessLevel'] = pd.cut(
                df_policy['ExcessSelected'],
                bins=[0, excess_quantiles[0.33], excess_quantiles[0.67], 
                      df_policy['ExcessSelected'].max()],
                labels=['Low_Excess', 'Medium_Excess', 'High_Excess']
            )
        
        return df_policy
    
    def create_interaction_features(self, df, feature_pairs=None):
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples indicating which features to interact
        """
        df_interactions = df.copy()
        
        if feature_pairs is None:
            # Default important interactions for insurance
            feature_pairs = [
                ('VehicleAge', 'Kilowatts'),
                ('CustomValueEstimate', 'VehicleAge'),
                ('TotalPremium', 'SumInsured'),
                ('SafetyScore', 'VehicleAge')  # If these features exist
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_interactions.columns and feat2 in df_interactions.columns:
                # Create multiplicative interaction
                df_interactions[f'{feat1}_x_{feat2}'] = df_interactions[feat1] * df_interactions[feat2]
                
                # Create ratio interaction (if feat2 is not zero)
                df_interactions[f'{feat1}_div_{feat2}'] = df_interactions[feat1] / (df_interactions[feat2] + 1e-8)
        
        return df_interactions
    
    def select_features(self, X, y, method='mutual_info', k=20, problem_type='regression'):
        """
        Select the most important features using various methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            k: Number of features to select
            problem_type: 'regression' or 'classification'
        """
        if method == 'mutual_info':
            if problem_type == 'regression':
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
            else:
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        elif method == 'f_test':
            if problem_type == 'regression':
                selector = SelectKBest(score_func=f_regression, k=k)
            else:
                selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selectors[method] = {
            'selector': selector,
            'selected_features': selected_features,
            'scores': selector.scores_
        }
        
        self.logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return X_selected, selected_features
    
    def create_pca_features(self, X, n_components=10):
        """
        Create principal component features for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of principal components
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame with PCA features
        pca_columns = [f'PC_{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        self.feature_selectors['pca'] = {
            'pca': pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_zzvariance': np.cumsum(pca.explained_variance_ratio_)
        }
        
        return X_pca_df