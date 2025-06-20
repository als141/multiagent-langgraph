"""Visualization tools for multi-agent experiment analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
from dataclasses import dataclass

# Import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import networkx as nx
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

from ..utils import get_logger
from .analysis_engine import AnalysisResults

logger = get_logger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization styling and parameters."""
    
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = "pdf"
    interactive: bool = True
    
    # Color schemes for different plot types
    cooperation_colors: List[str] = None
    strategy_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.cooperation_colors is None:
            self.cooperation_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
        
        if self.strategy_colors is None:
            self.strategy_colors = {
                "tit_for_tat": "#1f77b4",
                "always_cooperate": "#2ca02c", 
                "always_defect": "#d62728",
                "adaptive_tit_for_tat": "#ff7f0e",
                "pavlov": "#9467bd",
                "evolutionary": "#8c564b",
                "random": "#e377c2",
                "generous_tit_for_tat": "#7f7f7f",
                "suspicious_tit_for_tat": "#bcbd22"
            }


class ExperimentVisualizer:
    """Main visualization engine for multi-agent experiments."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration parameters
        """
        
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style
        try:
            plt.style.use(self.config.style)
        except OSError:
            plt.style.use('default')
            logger.warning(f"Style '{self.config.style}' not available, using default")
        
        # Set color palette
        sns.set_palette(self.config.color_palette)
        
        logger.debug("Experiment visualizer initialized", 
                    style=self.config.style, 
                    interactive=self.config.interactive)
    
    def create_experiment_dashboard(
        self,
        experiment_data: pd.DataFrame,
        analysis_results: AnalysisResults,
        output_dir: Path
    ) -> Dict[str, str]:
        """Create comprehensive visualization dashboard.
        
        Args:
            experiment_data: Experimental results DataFrame
            analysis_results: Statistical analysis results
            output_dir: Directory for saving plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        
        logger.info("Creating experiment dashboard", data_shape=experiment_data.shape)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # 1. Cooperation evolution over time
        plots["cooperation_timeline"] = self._plot_cooperation_timeline(
            experiment_data, output_dir / "cooperation_timeline"
        )
        
        # 2. Strategy distribution analysis
        plots["strategy_distribution"] = self._plot_strategy_distribution(
            experiment_data, output_dir / "strategy_distribution"
        )
        
        # 3. Payoff analysis
        plots["payoff_analysis"] = self._plot_payoff_analysis(
            experiment_data, output_dir / "payoff_analysis"
        )
        
        # 4. Statistical test results
        plots["statistical_results"] = self._plot_statistical_results(
            analysis_results, output_dir / "statistical_results"
        )
        
        # 5. Correlation heatmap
        plots["correlation_heatmap"] = self._plot_correlation_heatmap(
            experiment_data, output_dir / "correlation_heatmap"
        )
        
        # 6. Experimental conditions comparison
        plots["conditions_comparison"] = self._plot_conditions_comparison(
            experiment_data, output_dir / "conditions_comparison"
        )
        
        # 7. Time series trends
        plots["time_series_trends"] = self._plot_time_series_trends(
            experiment_data, output_dir / "time_series_trends"
        )
        
        # 8. Network visualization (if network data available)
        if self._has_network_data(experiment_data):
            plots["network_analysis"] = self._plot_network_analysis(
                experiment_data, output_dir / "network_analysis"
            )
        
        # Create interactive dashboard if Plotly available
        if PLOTLY_AVAILABLE and self.config.interactive:
            plots["interactive_dashboard"] = self._create_interactive_dashboard(
                experiment_data, analysis_results, output_dir / "interactive_dashboard.html"
            )
        
        logger.info("Dashboard created", plots_count=len(plots))
        
        return plots
    
    def _plot_cooperation_timeline(self, data: pd.DataFrame, output_path: Path) -> str:
        """Plot cooperation rate evolution over time."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Cooperation Evolution Analysis", fontsize=16, fontweight='bold')
        
        # 1. Overall cooperation trend
        cooperation_cols = [col for col in data.columns if 'cooperation_rate' in col]
        if cooperation_cols:
            ax = axes[0, 0]
            for col in cooperation_cols[:3]:  # Limit to avoid clutter
                if data[col].notna().sum() > 0:
                    values = data[col].dropna()
                    ax.plot(values.index, values, label=col.replace('_', ' ').title(), 
                           linewidth=2, alpha=0.8)
            ax.set_title("Cooperation Rate Trends")
            ax.set_xlabel("Experimental Run")
            ax.set_ylabel("Cooperation Rate")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Distribution by experimental conditions
        ax = axes[0, 1]
        condition_cols = [col for col in data.columns if col.startswith('condition_')]
        if condition_cols and cooperation_cols:
            main_coop_col = cooperation_cols[0]
            main_condition = condition_cols[0]
            
            if data[main_coop_col].notna().sum() > 0 and main_condition in data.columns:
                sns.boxplot(data=data, x=main_condition, y=main_coop_col, ax=ax)
                ax.set_title(f"Cooperation by {main_condition.replace('condition_', '').title()}")
                ax.set_ylabel("Cooperation Rate")
        
        # 3. Final vs Initial cooperation rates
        ax = axes[1, 0]
        final_coop_cols = [col for col in data.columns if 'final_cooperation' in col]
        if final_coop_cols:
            final_col = final_coop_cols[0]
            if data[final_col].notna().sum() > 0:
                ax.hist(data[final_col].dropna(), bins=20, alpha=0.7, 
                       color=self.config.cooperation_colors[2])
                ax.set_title("Final Cooperation Rate Distribution")
                ax.set_xlabel("Final Cooperation Rate")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
        
        # 4. Cooperation trend analysis
        ax = axes[1, 1]
        trend_cols = [col for col in data.columns if 'trend' in col and 'cooperation' in col]
        if trend_cols:
            trend_col = trend_cols[0]
            if data[trend_col].notna().sum() > 0:
                trend_data = data[trend_col].dropna()
                colors = ['red' if x < 0 else 'green' for x in trend_data]
                ax.scatter(range(len(trend_data)), trend_data, c=colors, alpha=0.6)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_title("Cooperation Trend Analysis")
                ax.set_xlabel("Experimental Run")
                ax.set_ylabel("Trend Slope")
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_strategy_distribution(self, data: pd.DataFrame, output_path: Path) -> str:
        """Plot strategy distribution and evolution."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Strategy Distribution Analysis", fontsize=16, fontweight='bold')
        
        # Look for strategy-related columns
        strategy_cols = [col for col in data.columns if 'strategy' in col.lower()]
        
        if not strategy_cols:
            # Create placeholder plot
            axes[0, 0].text(0.5, 0.5, "No strategy data available", 
                          ha='center', va='center', transform=axes[0, 0].transAxes)
            plt.tight_layout()
            output_file = f"{output_path}.{self.config.save_format}"
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return output_file
        
        # 1. Strategy diversity over time
        ax = axes[0, 0]
        diversity_cols = [col for col in data.columns if 'diversity' in col and 'strategy' in col]
        if diversity_cols:
            diversity_col = diversity_cols[0]
            if data[diversity_col].notna().sum() > 0:
                values = data[diversity_col].dropna()
                ax.plot(values.index, values, linewidth=2, color=self.config.cooperation_colors[1])
                ax.set_title("Strategy Diversity Evolution")
                ax.set_xlabel("Experimental Run")
                ax.set_ylabel("Shannon Diversity Index")
                ax.grid(True, alpha=0.3)
        
        # 2. Strategy performance comparison
        ax = axes[0, 1]
        payoff_cols = [col for col in data.columns if 'payoff' in col]
        if payoff_cols and len(data) > 0:
            payoff_col = payoff_cols[0]
            condition_cols = [col for col in data.columns if col.startswith('condition_')]
            if condition_cols and data[payoff_col].notna().sum() > 0:
                condition_col = condition_cols[0]
                sns.boxplot(data=data, x=condition_col, y=payoff_col, ax=ax)
                ax.set_title("Strategy Performance by Condition")
                ax.set_ylabel("Payoff")
        
        # 3. Strategy survival analysis
        ax = axes[1, 0]
        # This would show which strategies persist over time
        ax.text(0.5, 0.5, "Strategy Survival Analysis\n(Requires longitudinal data)", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Strategy Persistence")
        
        # 4. Strategy interaction matrix
        ax = axes[1, 1]
        # Show how different strategies perform against each other
        ax.text(0.5, 0.5, "Strategy Interaction Matrix\n(Requires pairwise data)", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Strategy Interactions")
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_payoff_analysis(self, data: pd.DataFrame, output_path: Path) -> str:
        """Plot payoff distribution and analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Payoff Analysis", fontsize=16, fontweight='bold')
        
        payoff_cols = [col for col in data.columns if 'payoff' in col]
        
        if not payoff_cols:
            axes[0, 0].text(0.5, 0.5, "No payoff data available", 
                          ha='center', va='center', transform=axes[0, 0].transAxes)
            plt.tight_layout()
            output_file = f"{output_path}.{self.config.save_format}"
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return output_file
        
        # 1. Payoff distribution
        ax = axes[0, 0]
        main_payoff_col = payoff_cols[0]
        if data[main_payoff_col].notna().sum() > 0:
            values = data[main_payoff_col].dropna()
            ax.hist(values, bins=20, alpha=0.7, color=self.config.cooperation_colors[3])
            ax.axvline(values.mean(), color='red', linestyle='--', 
                      label=f'Mean: {values.mean():.3f}')
            ax.axvline(values.median(), color='orange', linestyle='--', 
                      label=f'Median: {values.median():.3f}')
            ax.set_title("Payoff Distribution")
            ax.set_xlabel("Payoff")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Payoff vs Cooperation relationship
        ax = axes[0, 1]
        cooperation_cols = [col for col in data.columns if 'cooperation_rate' in col]
        if cooperation_cols and data[main_payoff_col].notna().sum() > 0:
            coop_col = cooperation_cols[0]
            if data[coop_col].notna().sum() > 0:
                valid_data = data[[main_payoff_col, coop_col]].dropna()
                if len(valid_data) > 0:
                    ax.scatter(valid_data[coop_col], valid_data[main_payoff_col], 
                             alpha=0.6, color=self.config.cooperation_colors[1])
                    
                    # Add trend line
                    try:
                        z = np.polyfit(valid_data[coop_col], valid_data[main_payoff_col], 1)
                        p = np.poly1d(z)
                        ax.plot(valid_data[coop_col], p(valid_data[coop_col]), 
                               "r--", alpha=0.8, linewidth=2)
                    except:
                        pass
                    
                    ax.set_title("Payoff vs Cooperation Rate")
                    ax.set_xlabel("Cooperation Rate")
                    ax.set_ylabel("Payoff")
                    ax.grid(True, alpha=0.3)
        
        # 3. Payoff inequality analysis
        ax = axes[1, 0]
        inequality_cols = [col for col in data.columns if 'inequality' in col]
        if inequality_cols:
            inequality_col = inequality_cols[0]
            if data[inequality_col].notna().sum() > 0:
                values = data[inequality_col].dropna()
                ax.plot(values.index, values, linewidth=2, 
                       color=self.config.cooperation_colors[0])
                ax.set_title("Payoff Inequality Over Time")
                ax.set_xlabel("Experimental Run")
                ax.set_ylabel("Inequality Index")
                ax.grid(True, alpha=0.3)
        
        # 4. Payoff by experimental conditions
        ax = axes[1, 1]
        condition_cols = [col for col in data.columns if col.startswith('condition_')]
        if condition_cols and data[main_payoff_col].notna().sum() > 0:
            condition_col = condition_cols[0]
            if data[condition_col].notna().sum() > 0:
                sns.violinplot(data=data, x=condition_col, y=main_payoff_col, ax=ax)
                ax.set_title(f"Payoff by {condition_col.replace('condition_', '').title()}")
                ax.set_ylabel("Payoff")
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_statistical_results(self, analysis_results: AnalysisResults, output_path: Path) -> str:
        """Visualize statistical test results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Statistical Analysis Results", fontsize=16, fontweight='bold')
        
        # 1. P-values visualization
        ax = axes[0, 0]
        if analysis_results.statistical_tests:
            test_names = list(analysis_results.statistical_tests.keys())
            p_values = [test.p_value for test in analysis_results.statistical_tests.values()]
            
            # Truncate long test names
            display_names = [name[:20] + '...' if len(name) > 20 else name for name in test_names]
            
            colors = ['red' if p < 0.05 else 'blue' for p in p_values]
            bars = ax.barh(display_names, p_values, color=colors, alpha=0.7)
            ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
            ax.set_title("Statistical Test P-Values")
            ax.set_xlabel("P-Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add significance annotations
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       significance, va='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No statistical test results available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 2. Effect sizes
        ax = axes[0, 1]
        if analysis_results.statistical_tests:
            tests_with_effects = {name: test for name, test in analysis_results.statistical_tests.items() 
                                if test.effect_size is not None}
            
            if tests_with_effects:
                names = list(tests_with_effects.keys())
                effect_sizes = [test.effect_size for test in tests_with_effects.values()]
                display_names = [name[:20] + '...' if len(name) > 20 else name for name in names]
                
                colors = ['green' if abs(es) > 0.5 else 'yellow' if abs(es) > 0.3 else 'red' 
                         for es in effect_sizes]
                ax.barh(display_names, [abs(es) for es in effect_sizes], color=colors, alpha=0.7)
                ax.set_title("Effect Sizes")
                ax.set_xlabel("Effect Size (|d|)")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No effect size data available", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 3. Correlation significance
        ax = axes[1, 0]
        if analysis_results.correlation_analysis and 'significant_correlations' in analysis_results.correlation_analysis:
            correlations = analysis_results.correlation_analysis['significant_correlations'][:10]  # Top 10
            
            if correlations:
                corr_values = [corr['correlation'] for corr in correlations]
                labels = [f"{corr['variable1'][:10]}...\nvs\n{corr['variable2'][:10]}..." 
                         for corr in correlations]
                
                colors = ['darkred' if abs(r) > 0.7 else 'red' if abs(r) > 0.5 else 'orange' 
                         for r in corr_values]
                ax.barh(labels, [abs(r) for r in corr_values], color=colors, alpha=0.7)
                ax.set_title("Top Correlations")
                ax.set_xlabel("Correlation Strength |r|")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No significant correlations found", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 4. Model performance (R²)
        ax = axes[1, 1]
        if analysis_results.regression_analysis:
            models = analysis_results.regression_analysis
            model_names = list(models.keys())
            r_squared_values = [model['r_squared'] for model in models.values()]
            
            display_names = [name[:15] + '...' if len(name) > 15 else name for name in model_names]
            colors = ['green' if r2 > 0.5 else 'yellow' if r2 > 0.3 else 'red' for r2 in r_squared_values]
            
            ax.barh(display_names, r_squared_values, color=colors, alpha=0.7)
            ax.set_title("Regression Model Performance")
            ax.set_xlabel("R² Score")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No regression analysis available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_correlation_heatmap(self, data: pd.DataFrame, output_path: Path) -> str:
        """Create correlation heatmap of key variables."""
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, "Insufficient numeric variables for correlation analysis", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Correlation Heatmap")
            plt.tight_layout()
            output_file = f"{output_path}.{self.config.save_format}"
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return output_file
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(min(16, len(correlation_matrix.columns)), 
                                      min(16, len(correlation_matrix.columns))))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Hide upper triangle
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   center=0, cmap='RdBu_r', square=True, ax=ax,
                   cbar_kws={"shrink": .8})
        
        ax.set_title("Variable Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_conditions_comparison(self, data: pd.DataFrame, output_path: Path) -> str:
        """Compare results across experimental conditions."""
        
        condition_cols = [col for col in data.columns if col.startswith('condition_')]
        
        if not condition_cols:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, "No experimental conditions found", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Experimental Conditions Comparison")
            plt.tight_layout()
            output_file = f"{output_path}.{self.config.save_format}"
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return output_file
        
        # Create subplots for different metrics
        n_conditions = min(len(condition_cols), 4)  # Limit to 4 conditions
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Experimental Conditions Comparison", fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Key metrics to compare
        comparison_metrics = [
            'metric_mean_cooperation_rate',
            'metric_mean_payoff', 
            'summary_final_cooperation_rate',
            'summary_mean_payoff'
        ]
        
        for i, condition_col in enumerate(condition_cols[:n_conditions]):
            ax = axes[i]
            
            # Find a suitable metric for this condition
            available_metrics = [m for m in comparison_metrics if m in data.columns]
            if available_metrics:
                metric_col = available_metrics[0]
                
                if data[metric_col].notna().sum() > 0 and data[condition_col].notna().sum() > 0:
                    sns.boxplot(data=data, x=condition_col, y=metric_col, ax=ax)
                    ax.set_title(f"{metric_col.replace('_', ' ').title()}\nby {condition_col.replace('condition_', '').title()}")
                    ax.set_xlabel(condition_col.replace('condition_', '').title())
                    ax.set_ylabel(metric_col.replace('_', ' ').title())
                else:
                    ax.text(0.5, 0.5, f"No data for {condition_col}", 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No suitable metrics found", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide unused subplots
        for i in range(n_conditions, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_time_series_trends(self, data: pd.DataFrame, output_path: Path) -> str:
        """Plot time series trends and patterns."""
        
        # Look for time series data
        time_series_cols = [col for col in data.columns if 'trend' in col or 'series' in col]
        metric_cols = [col for col in data.columns if col.startswith('metric_')]
        
        if not time_series_cols and not metric_cols:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, "No time series data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Time Series Trends")
            plt.tight_layout()
            output_file = f"{output_path}.{self.config.save_format}"
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return output_file
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Time Series Analysis", fontsize=16, fontweight='bold')
        
        # 1. Trend analysis
        ax = axes[0, 0]
        trend_cols = [col for col in data.columns if 'trend' in col]
        if trend_cols:
            trend_col = trend_cols[0]
            if data[trend_col].notna().sum() > 0:
                values = data[trend_col].dropna()
                ax.hist(values, bins=20, alpha=0.7, color=self.config.cooperation_colors[1])
                ax.axvline(0, color='black', linestyle='--', alpha=0.5)
                ax.axvline(values.mean(), color='red', linestyle='-', 
                          label=f'Mean: {values.mean():.4f}')
                ax.set_title("Trend Distribution")
                ax.set_xlabel("Trend Slope")
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 2. Metric evolution
        ax = axes[0, 1]
        if metric_cols:
            for i, col in enumerate(metric_cols[:3]):  # Limit to 3 metrics
                if data[col].notna().sum() > 0:
                    values = data[col].dropna()
                    ax.plot(values.index, values, label=col.replace('metric_', '').replace('_', ' '), 
                           linewidth=2, alpha=0.8)
            ax.set_title("Metric Evolution")
            ax.set_xlabel("Experimental Run")
            ax.set_ylabel("Metric Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Seasonal/Cyclical patterns
        ax = axes[1, 0]
        ax.text(0.5, 0.5, "Seasonal Analysis\n(Requires temporal data)", 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Cyclical Patterns")
        
        # 4. Volatility analysis
        ax = axes[1, 1]
        volatility_cols = [col for col in data.columns if 'std' in col]
        if volatility_cols:
            vol_col = volatility_cols[0]
            if data[vol_col].notna().sum() > 0:
                values = data[vol_col].dropna()
                ax.plot(values.index, values, linewidth=2, 
                       color=self.config.cooperation_colors[0])
                ax.set_title("Volatility Analysis")
                ax.set_xlabel("Experimental Run")
                ax.set_ylabel("Standard Deviation")
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No volatility data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _has_network_data(self, data: pd.DataFrame) -> bool:
        """Check if network analysis data is available."""
        network_indicators = ['trust', 'network', 'centrality', 'connection']
        return any(indicator in ' '.join(data.columns).lower() for indicator in network_indicators)
    
    def _plot_network_analysis(self, data: pd.DataFrame, output_path: Path) -> str:
        """Plot network analysis visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Network Analysis", fontsize=16, fontweight='bold')
        
        # Network visualization would require actual network data
        # This is a placeholder implementation
        
        for i, ax in enumerate(axes.flatten()):
            network_aspects = [
                "Trust Network Topology",
                "Connection Density Over Time", 
                "Centrality Measures",
                "Community Structure"
            ]
            
            ax.text(0.5, 0.5, f"{network_aspects[i]}\n(Requires network data)", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(network_aspects[i])
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _create_interactive_dashboard(
        self, 
        data: pd.DataFrame, 
        analysis_results: AnalysisResults, 
        output_path: Path
    ) -> str:
        """Create interactive Plotly dashboard."""
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, skipping interactive dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cooperation Timeline', 'Payoff Distribution',
                          'Strategy Distribution', 'Statistical Results',
                          'Correlation Network', 'Experimental Conditions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces for different visualizations
        cooperation_cols = [col for col in data.columns if 'cooperation_rate' in col]
        if cooperation_cols:
            col = cooperation_cols[0]
            if data[col].notna().sum() > 0:
                values = data[col].dropna()
                fig.add_trace(
                    go.Scatter(x=values.index, y=values, mode='lines+markers', 
                             name='Cooperation Rate'),
                    row=1, col=1
                )
        
        # Payoff distribution
        payoff_cols = [col for col in data.columns if 'payoff' in col]
        if payoff_cols:
            col = payoff_cols[0]
            if data[col].notna().sum() > 0:
                values = data[col].dropna()
                fig.add_trace(
                    go.Histogram(x=values, name='Payoff Distribution'),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Multi-Agent Experiment Interactive Dashboard",
            showlegend=True
        )
        
        # Save as HTML
        pyo.plot(fig, filename=str(output_path), auto_open=False)
        
        logger.info("Interactive dashboard created", output_file=str(output_path))
        
        return str(output_path)
    
    def create_publication_figures(
        self,
        experiment_data: pd.DataFrame,
        analysis_results: AnalysisResults,
        output_dir: Path,
        figure_style: str = "publication"
    ) -> Dict[str, str]:
        """Create publication-ready figures with specific formatting.
        
        Args:
            experiment_data: Experimental results DataFrame
            analysis_results: Statistical analysis results
            output_dir: Directory for saving figures
            figure_style: Style for publication figures
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        
        # Set publication style
        if figure_style == "publication":
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'lines.linewidth': 2,
                'axes.linewidth': 1,
                'grid.linewidth': 0.5
            })
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        publication_figures = {}
        
        # Main results figure (for paper)
        publication_figures["main_results"] = self._create_main_results_figure(
            experiment_data, analysis_results, output_dir / "figure_1_main_results"
        )
        
        # Statistical comparison figure
        publication_figures["statistical_comparison"] = self._create_statistical_figure(
            analysis_results, output_dir / "figure_2_statistical_results"
        )
        
        # Time series evolution figure
        publication_figures["evolution_dynamics"] = self._create_evolution_figure(
            experiment_data, output_dir / "figure_3_evolution_dynamics"
        )
        
        logger.info("Publication figures created", figures_count=len(publication_figures))
        
        return publication_figures
    
    def _create_main_results_figure(self, data: pd.DataFrame, analysis: AnalysisResults, 
                                  output_path: Path) -> str:
        """Create main results figure for publication."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Main Experimental Results", fontsize=16, fontweight='bold')
        
        # Cooperation rates by condition
        ax = axes[0]
        cooperation_cols = [col for col in data.columns if 'cooperation_rate' in col]
        condition_cols = [col for col in data.columns if col.startswith('condition_')]
        
        if cooperation_cols and condition_cols:
            coop_col = cooperation_cols[0]
            condition_col = condition_cols[0]
            
            if data[coop_col].notna().sum() > 0 and data[condition_col].notna().sum() > 0:
                sns.boxplot(data=data, x=condition_col, y=coop_col, ax=ax)
                ax.set_title("Cooperation Rate by Condition")
                ax.set_ylabel("Cooperation Rate")
                ax.set_xlabel("Experimental Condition")
        
        # Payoff analysis
        ax = axes[1]
        payoff_cols = [col for col in data.columns if 'payoff' in col]
        if payoff_cols and cooperation_cols:
            payoff_col = payoff_cols[0]
            coop_col = cooperation_cols[0]
            
            valid_data = data[[payoff_col, coop_col]].dropna()
            if len(valid_data) > 0:
                ax.scatter(valid_data[coop_col], valid_data[payoff_col], alpha=0.6)
                ax.set_xlabel("Cooperation Rate")
                ax.set_ylabel("Payoff")
                ax.set_title("Payoff vs Cooperation")
        
        # Statistical significance
        ax = axes[2]
        if analysis.statistical_tests:
            test_names = list(analysis.statistical_tests.keys())[:5]  # Top 5 tests
            p_values = [analysis.statistical_tests[name].p_value for name in test_names]
            
            colors = ['red' if p < 0.05 else 'blue' for p in p_values]
            bars = ax.barh(test_names, p_values, color=colors, alpha=0.7)
            ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.8)
            ax.set_xlabel("P-Value")
            ax.set_title("Statistical Significance")
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _create_statistical_figure(self, analysis: AnalysisResults, output_path: Path) -> str:
        """Create statistical results figure."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("Statistical Analysis Results", fontsize=16, fontweight='bold')
        
        # Effect sizes
        ax = axes[0]
        if analysis.statistical_tests:
            tests_with_effects = {name: test for name, test in analysis.statistical_tests.items() 
                                if test.effect_size is not None}
            
            if tests_with_effects:
                names = list(tests_with_effects.keys())
                effect_sizes = [test.effect_size for test in tests_with_effects.values()]
                
                colors = ['green' if abs(es) > 0.5 else 'yellow' if abs(es) > 0.3 else 'red' 
                         for es in effect_sizes]
                ax.barh(names, [abs(es) for es in effect_sizes], color=colors, alpha=0.7)
                ax.set_xlabel("Effect Size |d|")
                ax.set_title("Effect Sizes")
        
        # Model performance
        ax = axes[1]
        if analysis.regression_analysis:
            models = analysis.regression_analysis
            model_names = list(models.keys())
            r_squared_values = [model['r_squared'] for model in models.values()]
            
            colors = ['green' if r2 > 0.5 else 'yellow' if r2 > 0.3 else 'red' 
                     for r2 in r_squared_values]
            ax.barh(model_names, r_squared_values, color=colors, alpha=0.7)
            ax.set_xlabel("R² Score")
            ax.set_title("Model Performance")
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _create_evolution_figure(self, data: pd.DataFrame, output_path: Path) -> str:
        """Create evolution dynamics figure."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot cooperation evolution over time
        cooperation_cols = [col for col in data.columns if 'cooperation_rate' in col]
        
        if cooperation_cols:
            for col in cooperation_cols[:3]:  # Limit to 3 metrics
                if data[col].notna().sum() > 0:
                    values = data[col].dropna()
                    ax.plot(values.index, values, 
                           label=col.replace('_', ' ').title(), 
                           linewidth=2, alpha=0.8)
        
        ax.set_title("Evolution of Cooperation Over Time", fontsize=16, fontweight='bold')
        ax.set_xlabel("Experimental Run")
        ax.set_ylabel("Cooperation Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f"{output_path}.{self.config.save_format}"
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file