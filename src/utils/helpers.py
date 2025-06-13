import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_loss_ratio(premiums: pd.Series, claims: pd.Series) -> pd.Series:
    """
    Calculate loss ratio (Claims/Premium) safely handling division by zero.
    
    Args:
        premiums: Series of premium values
        claims: Series of claim values
    
    Returns:
        Series of loss ratios
    """
    loss_ratio = claims / premiums
    return loss_ratio.replace([np.inf, -np.inf], np.nan)

def summarize_categorical(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """
    Summarize categorical column with counts and percentages.
    
    Args:
        df: DataFrame
        column: Column name to summarize
        top_n: Number of top categories to return
    
    Returns:
        DataFrame with summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    counts = df[column].value_counts().head(top_n)
    percentages = (counts / len(df)) * 100
    
    summary = pd.DataFrame({
        'Category': counts.index,
        'Count': counts.values,
        'Percentage': percentages.values
    })
    
    return summary

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers using IQR method.
    
    Args:
        series: Pandas Series to analyze
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        Dictionary with outlier information
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(series)) * 100,
        'outlier_indices': outliers.index.tolist()
    }

def create_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create and display correlation matrix for numerical columns.
    
    Args:
        df: DataFrame
        figsize: Figure size tuple
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.show()

def business_insights_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate key business insights from the insurance data.
    
    Args:
        df: Insurance DataFrame
    
    Returns:
        Dictionary with business insights
    """
    insights = {}
    
    # Overall portfolio metrics
    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        total_premium = df['TotalPremium'].sum()
        total_claims = df['TotalClaims'].sum()
        overall_loss_ratio = total_claims / total_premium
        
        insights['portfolio_metrics'] = {
            'total_premium': total_premium,
            'total_claims': total_claims,
            'overall_loss_ratio': overall_loss_ratio,
            'profit_margin': 1 - overall_loss_ratio,
            'total_policies': len(df)
        }
    
    # Risk by province
    if 'Province' in df.columns:
        province_risk = df.groupby('Province').agg({
            'TotalClaims': 'sum',
            'TotalPremium': 'sum',
            'PolicyID': 'count'
        })
        province_risk['LossRatio'] = province_risk['TotalClaims'] / province_risk['TotalPremium']
        insights['province_risk'] = province_risk.sort_values('LossRatio', ascending=False)
    
    # Risk by gender
    if 'Gender' in df.columns:
        gender_risk = df.groupby('Gender').agg({
            'TotalClaims': 'sum',
            'TotalPremium': 'sum',
            'PolicyID': 'count'
        })
        gender_risk['LossRatio'] = gender_risk['TotalClaims'] / gender_risk['TotalPremium']
        insights['gender_risk'] = gender_risk
    
    return insights