import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import joblib
import logging

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for insurance data.
    
    This class handles missing values, feature encoding, scaling, and data splitting
    with specific considerations for insurance domain requirements.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def handle_missing_values(self, df, strategy='mixed'):
        """
        Handle missing values with domain-specific strategies.
        
        Args:
            df: Input DataFrame
            strategy: 'mixed', 'simple', or 'knn'
        
        Returns:
            DataFrame with handled missing values
        """
        df_processed = df.copy()
        
        # Identify columns by type
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        if strategy == 'mixed':
            # For numeric columns with < 5% missing: median imputation
            # For numeric columns with > 5% missing: KNN imputation
            for col in numeric_columns:
                missing_pct = df_processed[col].isnull().sum() / len(df_processed)
                
                if missing_pct > 0 and missing_pct <= 0.05:
                    imputer = SimpleImputer(strategy='median')
                    df_processed[col] = imputer.fit_transform(df_processed[[col]]).flatten()
                    self.imputers[col] = imputer
                elif missing_pct > 0.05:
                    # Use KNN for high missing percentage
                    knn_imputer = KNNImputer(n_neighbors=5)
                    df_processed[col] = knn_imputer.fit_transform(df_processed[[col]]).flatten()
                    self.imputers[col] = knn_imputer
            
            # For categorical columns: mode imputation or 'Unknown' category
            for col in categorical_columns:
                missing_pct = df_processed[col].isnull().sum() / len(df_processed)
                if missing_pct > 0:
                    if missing_pct <= 0.10:
                        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                        df_processed[col].fillna(mode_value, inplace=True)
                    else:
                        df_processed[col].fillna('Unknown', inplace=True)
        
        self.logger.info(f"Missing values handled. Remaining missing: {df_processed.isnull().sum().sum()}")
        return df_processed
    
    def encode_categorical_features(self, df, encoding_strategy='mixed'):
        """
        Encode categorical features with appropriate strategies.
        
        Args:
            df: Input DataFrame
            encoding_strategy: 'onehot', 'label', or 'mixed'
        
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            unique_values = df_encoded[col].nunique()
            
            if encoding_strategy == 'mixed':
                # Use label encoding for high cardinality (>10), one-hot for low cardinality
                if unique_values <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(col, axis=1, inplace=True)
                    self.encoders[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
                else:
                    # Label encoding
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = {'type': 'label', 'encoder': le}
            
            elif encoding_strategy == 'onehot':
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
                self.encoders[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
        
        return df_encoded
    
    def scale_features(self, X_train, X_test, method='standard'):
        """
        Scale numerical features.
        
        Args:
            X_train, X_test: Training and test features
            method: 'standard', 'minmax', or 'robust'
        
        Returns:
            Scaled training and test sets
        """
        if method == 'standard':
            scaler = StandardScaler()
        
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Fit scaler on training data only
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        self.scalers['feature_scaler'] = scaler
        self.feature_names['numerical'] = numerical_cols.tolist()
        
        return X_train_scaled, X_test_scaled
    
    def prepare_claim_severity_data(self, df):
        """
        Prepare data specifically for claim severity modeling.
        Only includes records where TotalClaims > 0.
        """
        # Filter to only policies with claims
        claim_data = df[df['TotalClaims'] > 0].copy()
        
        self.logger.info(f"Claim severity dataset: {len(claim_data)} records with claims")
        
        # Handle missing values
        claim_data = self.handle_missing_values(claim_data)
        
        # Feature engineering for claim severity
        claim_data = self._engineer_claim_features(claim_data)
        
        return claim_data
    
    def prepare_premium_optimization_data(self, df):
        """
        Prepare data for premium optimization modeling.
        Includes all records.
        """
        premium_data = df.copy()
        
        # Handle missing values
        premium_data = self.handle_missing_values(premium_data)
        
        # Feature engineering for premium optimization
        premium_data = self._engineer_premium_features(premium_data)
        
        return premium_data
    
    def _engineer_claim_features(self, df):
        """Engineer features specific to claim severity prediction."""
        df_engineered = df.copy()
        
        # Vehicle age
        if 'RegistrationYear' in df_engineered.columns:
            current_year = df_engineered['TransactionMonth'].str[:4].astype(int).max()
            df_engineered['VehicleAge'] = current_year - df_engineered['RegistrationYear']
        
        # Claim to premium ratio (but we won't use this as a feature since it's leakage)
        # df_engineered['ClaimToPremiumRatio'] = df_engineered['TotalClaims'] / df_engineered['TotalPremium']
        
        # Vehicle value per kilowatt
        if 'CustomValueEstimate' in df_engineered.columns and 'Kilowatts' in df_engineered.columns:
            df_engineered['ValuePerKilowatt'] = df_engineered['CustomValueEstimate'] / (df_engineered['Kilowatts'] + 1)
        
        # Is high value vehicle
        if 'CustomValueEstimate' in df_engineered.columns:
            value_threshold = df_engineered['CustomValueEstimate'].quantile(0.75)
            df_engineered['IsHighValueVehicle'] = (df_engineered['CustomValueEstimate'] > value_threshold).astype(int)
        
        return df_engineered
    
    def _engineer_premium_features(self, df):
        """Engineer features specific to premium optimization."""
        df_engineered = df.copy()
        
        # Create claim indicator
        df_engineered['HasClaim'] = (df_engineered['TotalClaims'] > 0).astype(int)
        
        # Vehicle age
        if 'RegistrationYear' in df_engineered.columns:
            current_year = df_engineered['TransactionMonth'].str[:4].astype(int).max()
            df_engineered['VehicleAge'] = current_year - df_engineered['RegistrationYear']
        
        # Risk score based on vehicle characteristics
        df_engineered['RiskScore'] = 0
        
        # Add risk based on vehicle age
        if 'VehicleAge' in df_engineered.columns:
            df_engineered['RiskScore'] += df_engineered['VehicleAge'] * 0.1
        
        # Add risk based on vehicle value
        if 'CustomValueEstimate' in df_engineered.columns:
            value_normalized = df_engineered['CustomValueEstimate'] / df_engineered['CustomValueEstimate'].max()
            df_engineered['RiskScore'] += value_normalized * 0.3
        
        return df_engineered
    
    def split_data(self, X, y, test_size=0.2, stratify=None):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, 
                              random_state=self.random_state, 
                              stratify=stratify)
    
    def save_preprocessors(self, filepath):
        """Save all preprocessing objects for future use."""
        preprocessing_objects = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names
        }
        joblib.dump(preprocessing_objects, filepath)
        self.logger.info(f"Preprocessing objects saved to {filepath}")