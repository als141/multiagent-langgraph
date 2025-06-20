"""Statistical analysis engine for multi-agent experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import networkx as nx

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    significant: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "interpretation": self.interpretation,
            "significant": self.significant
        }


@dataclass
class AnalysisResults:
    """Container for complete analysis results."""
    
    experiment_id: str
    descriptive_stats: Dict[str, Any]
    statistical_tests: Dict[str, StatisticalResult]
    correlation_analysis: Dict[str, Any]
    regression_analysis: Dict[str, Any]
    network_analysis: Dict[str, Any]
    time_series_analysis: Dict[str, Any]
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save analysis results to JSON file."""
        
        filepath = Path(filepath)
        
        # Convert to serializable format
        serializable_data = {
            "experiment_id": self.experiment_id,
            "descriptive_stats": self.descriptive_stats,
            "statistical_tests": {
                name: result.to_dict() 
                for name, result in self.statistical_tests.items()
            },
            "correlation_analysis": self.correlation_analysis,
            "regression_analysis": self.regression_analysis,
            "network_analysis": self.network_analysis,
            "time_series_analysis": self.time_series_analysis
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("Analysis results saved", filepath=str(filepath))


class AnalysisEngine:
    """Statistical analysis engine for experimental data."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize analysis engine.
        
        Args:
            alpha: Significance level for statistical tests
        """
        
        self.alpha = alpha
        logger.debug("Analysis engine initialized", alpha=alpha)
    
    def analyze_experiment(
        self,
        experiment_data: pd.DataFrame,
        experiment_id: str,
        config: Dict[str, Any] = None
    ) -> AnalysisResults:
        """Perform comprehensive analysis of experimental data.
        
        Args:
            experiment_data: DataFrame containing experimental results
            experiment_id: Unique experiment identifier
            config: Analysis configuration parameters
            
        Returns:
            Complete analysis results
        """
        
        logger.info("Starting comprehensive analysis", experiment_id=experiment_id)
        
        # Descriptive statistics
        descriptive_stats = self._calculate_descriptive_stats(experiment_data)
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(experiment_data, config)
        
        # Correlation analysis
        correlation_analysis = self._perform_correlation_analysis(experiment_data)
        
        # Regression analysis
        regression_analysis = self._perform_regression_analysis(experiment_data)
        
        # Network analysis (if applicable)
        network_analysis = self._perform_network_analysis(experiment_data)
        
        # Time series analysis
        time_series_analysis = self._perform_time_series_analysis(experiment_data)
        
        results = AnalysisResults(
            experiment_id=experiment_id,
            descriptive_stats=descriptive_stats,
            statistical_tests=statistical_tests,
            correlation_analysis=correlation_analysis,
            regression_analysis=regression_analysis,
            network_analysis=network_analysis,
            time_series_analysis=time_series_analysis
        )
        
        logger.info("Analysis completed", experiment_id=experiment_id)
        
        return results
    
    def _calculate_descriptive_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics for all numeric variables."""
        
        stats_dict = {}
        
        # Overall statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if data[col].notna().sum() > 0:  # Skip empty columns
                stats_dict[col] = {
                    "count": data[col].count(),
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "median": data[col].median(),
                    "q25": data[col].quantile(0.25),
                    "q75": data[col].quantile(0.75),
                    "skewness": data[col].skew(),
                    "kurtosis": data[col].kurtosis()
                }
        
        # Categorical variables
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if data[col].notna().sum() > 0:
                value_counts = data[col].value_counts()
                stats_dict[f"{col}_categories"] = {
                    "unique_count": data[col].nunique(),
                    "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_common_freq": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    "distribution": value_counts.to_dict()
                }
        
        # Experimental conditions analysis
        condition_columns = [col for col in data.columns if col.startswith('condition_')]
        if condition_columns:
            stats_dict["experimental_conditions"] = {}
            for col in condition_columns:
                if data[col].notna().sum() > 0:
                    stats_dict["experimental_conditions"][col] = data[col].value_counts().to_dict()
        
        logger.debug("Descriptive statistics calculated", variables_count=len(stats_dict))
        
        return stats_dict
    
    def _perform_statistical_tests(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, StatisticalResult]:
        """Perform statistical hypothesis tests."""
        
        tests = {}
        
        # Configuration for tests
        if not config:
            config = {}
        
        # Primary dependent variables
        dependent_vars = config.get('dependent_variables', [
            'metric_mean_cooperation_rate',
            'metric_mean_payoff',
            'summary_final_cooperation_rate'
        ])
        
        # Independent variables (experimental factors)
        factor_columns = [col for col in data.columns if col.startswith('condition_')]
        
        # Perform tests for each dependent variable
        for dep_var in dependent_vars:
            if dep_var not in data.columns:
                continue
            
            # Skip if insufficient data
            if data[dep_var].notna().sum() < 3:
                continue
            
            # Test each experimental factor
            for factor in factor_columns:
                if factor not in data.columns:
                    continue
                
                factor_levels = data[factor].unique()
                if len(factor_levels) < 2:
                    continue
                
                test_name = f"{dep_var}_by_{factor}"
                
                try:
                    if len(factor_levels) == 2:
                        # Two-group comparison
                        group1 = data[data[factor] == factor_levels[0]][dep_var].dropna()
                        group2 = data[data[factor] == factor_levels[1]][dep_var].dropna()
                        
                        if len(group1) > 0 and len(group2) > 0:
                            # Check normality (simplified)
                            if len(group1) > 8 and len(group2) > 8:
                                # Use t-test for larger samples
                                statistic, p_value = ttest_ind(group1, group2)
                                test_type = "t_test"
                                
                                # Calculate effect size (Cohen's d)
                                pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                                    (len(group2) - 1) * group2.var()) / 
                                                   (len(group1) + len(group2) - 2))
                                effect_size = (group1.mean() - group2.mean()) / pooled_std
                            else:
                                # Use Mann-Whitney U for smaller samples
                                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                                test_type = "mann_whitney_u"
                                effect_size = None
                    else:
                        # Multi-group comparison
                        groups = [data[data[factor] == level][dep_var].dropna() for level in factor_levels]
                        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                        
                        if len(groups) >= 2:
                            statistic, p_value = kruskal(*groups)
                            test_type = "kruskal_wallis"
                            effect_size = None
                    
                    # Create result
                    result = StatisticalResult(
                        test_name=test_type,
                        statistic=statistic,
                        p_value=p_value,
                        effect_size=effect_size,
                        significant=p_value < self.alpha,
                        interpretation=self._interpret_test_result(test_type, p_value, effect_size)
                    )
                    
                    tests[test_name] = result
                    
                except Exception as e:
                    logger.warning(f"Statistical test failed: {test_name}", error=str(e))
        
        # Interaction effects (if multiple factors)
        if len(factor_columns) >= 2:
            tests.update(self._test_interactions(data, dependent_vars, factor_columns))
        
        logger.debug("Statistical tests completed", tests_count=len(tests))
        
        return tests
    
    def _test_interactions(
        self,
        data: pd.DataFrame,
        dependent_vars: List[str],
        factors: List[str]
    ) -> Dict[str, StatisticalResult]:
        """Test for interaction effects between factors."""
        
        interaction_tests = {}
        
        # Two-way interactions
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                factor1, factor2 = factors[i], factors[j]
                
                for dep_var in dependent_vars:
                    if dep_var not in data.columns:
                        continue
                    
                    try:
                        # Create interaction groups
                        interaction_groups = {}
                        for f1_val in data[factor1].unique():
                            for f2_val in data[factor2].unique():
                                mask = (data[factor1] == f1_val) & (data[factor2] == f2_val)
                                group_data = data[mask][dep_var].dropna()
                                if len(group_data) > 0:
                                    interaction_groups[f"{f1_val}_{f2_val}"] = group_data
                        
                        if len(interaction_groups) >= 2:
                            # Perform Kruskal-Wallis test on interaction groups
                            groups = list(interaction_groups.values())
                            statistic, p_value = kruskal(*groups)
                            
                            test_name = f"{dep_var}_interaction_{factor1}_{factor2}"
                            result = StatisticalResult(
                                test_name="interaction_test",
                                statistic=statistic,
                                p_value=p_value,
                                significant=p_value < self.alpha,
                                interpretation=f"Interaction between {factor1} and {factor2}"
                            )
                            
                            interaction_tests[test_name] = result
                    
                    except Exception as e:
                        logger.warning(f"Interaction test failed: {factor1} x {factor2}", error=str(e))
        
        return interaction_tests
    
    def _perform_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis between variables."""
        
        # Select numeric variables
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {"message": "Insufficient numeric variables for correlation analysis"}
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Find significant correlations
        significant_correlations = []
        n = len(numeric_data)
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                r = correlation_matrix.iloc[i, j]
                
                if not np.isnan(r) and abs(r) > 0.1:  # Minimum correlation threshold
                    # Calculate significance
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if abs(t_stat) < np.inf else 0
                    
                    significant_correlations.append({
                        "variable1": var1,
                        "variable2": var2,
                        "correlation": r,
                        "p_value": p_value,
                        "significant": p_value < self.alpha,
                        "strength": self._interpret_correlation_strength(abs(r))
                    })
        
        # Sort by absolute correlation
        significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "significant_correlations": significant_correlations[:20],  # Top 20
            "total_correlations": len(significant_correlations)
        }
    
    def _perform_regression_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform regression analysis to identify predictors."""
        
        regression_results = {}
        
        # Target variables for regression
        target_vars = [
            'metric_mean_cooperation_rate',
            'metric_mean_payoff',
            'summary_final_cooperation_rate'
        ]
        
        # Predictor variables
        predictor_vars = [col for col in data.columns if 
                         col.startswith('condition_') or 
                         col.startswith('summary_') or 
                         col.startswith('metric_')]
        
        for target in target_vars:
            if target not in data.columns:
                continue
            
            # Prepare data
            y = data[target].dropna()
            if len(y) < 10:  # Minimum sample size
                continue
            
            # Select predictors (excluding target and highly correlated variables)
            available_predictors = [p for p in predictor_vars if p != target and p in data.columns]
            
            if not available_predictors:
                continue
            
            X_full = data[available_predictors]
            
            # Align indices
            common_idx = X_full.index.intersection(y.index)
            X = X_full.loc[common_idx].select_dtypes(include=[np.number])
            y_aligned = y.loc[common_idx]
            
            # Remove highly correlated predictors
            if X.shape[1] > 1:
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                high_corr_pairs = [(col, idx) for col in upper_triangle.columns 
                                 for idx in upper_triangle.index 
                                 if upper_triangle.loc[idx, col] > 0.9]
                
                to_drop = [pair[1] for pair in high_corr_pairs]
                X = X.drop(columns=to_drop, errors='ignore')
            
            if X.shape[1] == 0 or len(X) < 5:
                continue
            
            try:
                # Fit regression model
                model = LinearRegression()
                model.fit(X, y_aligned)
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                r2 = r2_score(y_aligned, y_pred)
                
                # Feature importance
                feature_importance = []
                for i, feature in enumerate(X.columns):
                    importance = abs(model.coef_[i]) if hasattr(model, 'coef_') else 0
                    feature_importance.append({
                        "feature": feature,
                        "coefficient": model.coef_[i] if hasattr(model, 'coef_') else 0,
                        "importance": importance
                    })
                
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                
                regression_results[target] = {
                    "r_squared": r2,
                    "intercept": model.intercept_,
                    "n_features": X.shape[1],
                    "n_samples": len(y_aligned),
                    "feature_importance": feature_importance[:10],  # Top 10
                    "model_interpretation": self._interpret_regression_model(r2, X.shape[1])
                }
                
            except Exception as e:
                logger.warning(f"Regression analysis failed for {target}", error=str(e))
        
        return regression_results
    
    def _perform_network_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform network analysis on agent interactions."""
        
        # This is a placeholder for network analysis
        # In a real implementation, you would analyze trust networks, 
        # communication patterns, etc.
        
        network_metrics = {
            "analysis_type": "placeholder",
            "message": "Network analysis would be implemented based on interaction data"
        }
        
        # If we had network data, we would calculate:
        # - Network density
        # - Clustering coefficient
        # - Path lengths
        # - Centrality measures
        # - Community detection
        
        return network_metrics
    
    def _perform_time_series_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform time series analysis on temporal data."""
        
        time_series_results = {}
        
        # Look for time series columns
        time_series_cols = [col for col in data.columns if 
                           'cooperation_rate' in col and 'trend' in col]
        
        for col in time_series_cols:
            if col in data.columns and data[col].notna().sum() > 0:
                values = data[col].dropna()
                
                if len(values) > 3:
                    time_series_results[col] = {
                        "mean_trend": values.mean(),
                        "std_trend": values.std(),
                        "positive_trends": (values > 0).sum(),
                        "negative_trends": (values < 0).sum(),
                        "stable_trends": (values == 0).sum(),
                        "trend_interpretation": self._interpret_trend_analysis(values)
                    }
        
        return time_series_results
    
    def _interpret_test_result(
        self,
        test_type: str,
        p_value: float,
        effect_size: Optional[float] = None
    ) -> str:
        """Generate interpretation for statistical test result."""
        
        significance = "significant" if p_value < self.alpha else "not significant"
        
        interpretation = f"The {test_type} result is {significance} (p = {p_value:.4f})"
        
        if effect_size is not None:
            if abs(effect_size) < 0.2:
                effect_desc = "small"
            elif abs(effect_size) < 0.5:
                effect_desc = "medium"
            else:
                effect_desc = "large"
            
            interpretation += f" with a {effect_desc} effect size (d = {effect_size:.3f})"
        
        return interpretation
    
    def _interpret_correlation_strength(self, r: float) -> str:
        """Interpret correlation strength."""
        
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def _interpret_regression_model(self, r2: float, n_features: int) -> str:
        """Interpret regression model quality."""
        
        if r2 < 0.1:
            quality = "poor"
        elif r2 < 0.3:
            quality = "weak"
        elif r2 < 0.5:
            quality = "moderate"
        elif r2 < 0.7:
            quality = "good"
        else:
            quality = "excellent"
        
        return f"Model has {quality} explanatory power (R² = {r2:.3f}) with {n_features} predictors"
    
    def _interpret_trend_analysis(self, trends: pd.Series) -> str:
        """Interpret trend analysis results."""
        
        mean_trend = trends.mean()
        
        if abs(mean_trend) < 0.001:
            return "Overall stable trend with no clear direction"
        elif mean_trend > 0:
            return f"Overall positive trend (mean slope = {mean_trend:.4f})"
        else:
            return f"Overall negative trend (mean slope = {mean_trend:.4f})"