"""
Exploratory Data Analysis for Football Statistics
Comprehensive analysis and visualization of player/team data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class FootballEDA:
    """Exploratory Data Analysis for football statistics"""
    
    def __init__(self, data: pd.DataFrame, output_dir: str = 'outputs'):
        """
        Initialize EDA class
        
        Args:
            data: DataFrame with football statistics
            output_dir: Directory to save outputs
        """
        self.data = data.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loaded data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
    
    def generate_summary_statistics(self):
        """Generate and display summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Basic info
        print("\nDataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percent': missing_pct[missing > 0]
        }).sort_values('Missing', ascending=False)
        
        if not missing_df.empty:
            print(missing_df.head(10))
        else:
            print("No missing values found")
        
        # Numeric columns summary
        print("\nNumeric Columns Summary:")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_cols].describe().T)
        
        # Save summary
        summary_path = self.output_dir / 'summary_statistics.txt'
        with open(summary_path, 'w') as f:
            f.write("FOOTBALL STATISTICS SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Shape: {self.data.shape}\n")
            f.write(f"\nColumns: {list(self.data.columns)}\n")
            f.write(f"\nData types:\n{self.data.dtypes}\n")
            f.write(f"\nMissing values:\n{missing_df.to_string()}\n")
            f.write(f"\nNumeric summary:\n{self.data[numeric_cols].describe().to_string()}\n")
        
        print(f"\nSummary saved to {summary_path}")
    
    def analyze_distributions(self):
        """Analyze distributions of key metrics"""
        print("\n" + "="*60)
        print("DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Identify numeric columns for analysis
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Key metrics to analyze
        key_metrics = [col for col in numeric_cols if any(
            keyword in col.lower() for keyword in 
            ['goals', 'assists', 'minutes', 'matches', 'age', 'rating']
        )][:6]  # Limit to 6 most relevant
        
        if not key_metrics:
            print("No key metrics found for distribution analysis")
            return
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(key_metrics):
            if idx < len(axes):
                # Remove outliers for better visualization
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = self.data[
                    (self.data[col] >= Q1 - 1.5*IQR) & 
                    (self.data[col] <= Q3 + 1.5*IQR)
                ][col]
                
                # Plot histogram with KDE
                axes[idx].hist(filtered_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = filtered_data.mean()
                median_val = filtered_data.median()
                axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                axes[idx].legend(fontsize=8)
        
        # Remove empty subplots
        for idx in range(len(key_metrics), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        dist_path = self.output_dir / 'distributions.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution plots saved to {dist_path}")
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            print("Not enough numeric columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Plot heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=False,
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Correlation Matrix of Football Statistics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        corr_path = self.output_dir / 'correlation_matrix.png'
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find top correlations
        print("\nTop 10 Positive Correlations:")
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1.0]  # Remove self-correlations
        top_corr = corr_pairs.sort_values(ascending=False).head(10)
        print(top_corr)
        
        print(f"\nCorrelation heatmap saved to {corr_path}")
    
    def top_performers_analysis(self, top_n: int = 10):
        """Analyze top performers"""
        print("\n" + "="*60)
        print(f"TOP {top_n} PERFORMERS ANALYSIS")
        print("="*60)
        
        # Identify player name and key metric columns
        name_cols = [col for col in self.data.columns if 
                     any(keyword in col.lower() for keyword in ['player', 'name'])]
        
        metric_cols = [col for col in self.data.select_dtypes(include=[np.number]).columns if 
                       any(keyword in col.lower() for keyword in ['goals', 'assists', 'rating'])]
        
        if not name_cols or not metric_cols:
            print("Could not identify player names or metrics")
            return
        
        name_col = name_cols[0]
        
        # Analyze top performers for each metric
        for metric in metric_cols[:3]:  # Top 3 metrics
            print(f"\nTop {top_n} by {metric}:")
            top_players = self.data.nlargest(top_n, metric)[[name_col, metric]]
            print(top_players.to_string(index=False))
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(top_players)), top_players[metric].values, color='steelblue')
            plt.yticks(range(len(top_players)), top_players[name_col].values)
            plt.xlabel(metric, fontsize=12)
            plt.title(f'Top {top_n} Players by {metric}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            metric_safe = metric.replace('/', '_').replace(' ', '_')
            plot_path = self.output_dir / f'top_{top_n}_{metric_safe}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def league_comparison(self):
        """Compare statistics across leagues"""
        print("\n" + "="*60)
        print("LEAGUE COMPARISON")
        print("="*60)
        
        league_col = [col for col in self.data.columns if 'league' in col.lower()]
        
        if not league_col:
            print("No league column found for comparison")
            return
        
        league_col = league_col[0]
        
        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        key_metrics = [col for col in numeric_cols if 
                       any(keyword in col.lower() for keyword in ['goals', 'assists', 'rating'])][:3]
        
        if not key_metrics:
            print("No metrics found for league comparison")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(15, 5))
        if len(key_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(key_metrics):
            # Box plot by league
            self.data.boxplot(column=metric, by=league_col, ax=axes[idx])
            axes[idx].set_title(f'{metric} by League')
            axes[idx].set_xlabel('League')
            axes[idx].set_ylabel(metric)
            plt.sca(axes[idx])
            plt.xticks(rotation=45, ha='right')
        
        plt.suptitle('League Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        league_path = self.output_dir / 'league_comparison.png'
        plt.savefig(league_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nLeague comparison saved to {league_path}")
        
        # Print statistics by league
        print("\nAverage metrics by league:")
        league_stats = self.data.groupby(league_col)[key_metrics].mean()
        print(league_stats)
    
    def generate_full_report(self):
        """Generate complete EDA report"""
        print("\n" + "="*60)
        print("GENERATING FULL EDA REPORT")
        print("="*60)
        
        self.generate_summary_statistics()
        self.analyze_distributions()
        self.correlation_analysis()
        self.top_performers_analysis()
        self.league_comparison()
        
        print("\n" + "="*60)
        print(f"All outputs saved to: {self.output_dir}")
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Load sample data
    try:
        data = pd.read_csv('Big5_2020_21_Cleaned.csv')
        
        # Initialize EDA
        eda = FootballEDA(data)
        
        # Run full analysis
        eda.generate_full_report()
        
    except FileNotFoundError:
        print("Data file not found. Please ensure Big5_2020_21_Cleaned.csv exists in the directory.")
