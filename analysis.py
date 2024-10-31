"""
NYC Development Site Market Analysis
Analyzes the impact of Federal Reserve rate changes on NYC development site market (2018-2024).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load and prepare the data files."""
    # Define data paths
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Load the FED-RATES data
    fed_rates_df = pd.read_csv(data_dir / 'FED-RATES.csv')
    fed_rates_df['Date'] = pd.to_datetime(fed_rates_df['Date'])

    # Load the TRANSACTIONS-PT data
    transactions_df = pd.read_csv(data_dir / 'TRANSACTIONS-PT.csv')
    transactions_df['DATE'] = pd.to_datetime(transactions_df['DATE'], format='%m/%d/%Y')
    
    return fed_rates_df, transactions_df, output_dir

def expand_rate_periods(fed_rates_df, latest_date):
    """Create a continuous daily series of rates."""
    expanded_rates = pd.DataFrame()
    
    # Expand the rates into periods
    for i in range(len(fed_rates_df) - 1):
        start_date = fed_rates_df['Date'].iloc[i]
        end_date = fed_rates_df['Date'].iloc[i + 1] if i + 1 < len(fed_rates_df) else latest_date
        rate = fed_rates_df['New Rate (%)'].iloc[i]
        
        period_df = pd.DataFrame({
            'Date': pd.date_range(start=start_date, end=end_date, freq='D')[1:],
            'Rate': rate
        })
        expanded_rates = pd.concat([expanded_rates, period_df], ignore_index=True)
    
    # Add remaining dates if needed
    if fed_rates_df['Date'].iloc[-1] < latest_date:
        final_rate = fed_rates_df['New Rate (%)'].iloc[-1]
        remaining_period = pd.DataFrame({
            'Date': pd.date_range(
                start=fed_rates_df['Date'].iloc[-1] + pd.Timedelta(days=1),
                end=latest_date,
                freq='D'
            ),
            'Rate': final_rate
        })
        expanded_rates = pd.concat([expanded_rates, remaining_period], ignore_index=True)
    
    return expanded_rates

def calculate_metrics(merged_df):
    """Calculate various market metrics."""
    # Borough-level analysis
    borough_metrics = merged_df.groupby('BOROUGH').agg({
        'PPZFA': ['mean', 'median', 'std', 'count'],
        'PRICE': ['sum', 'mean'],
        'Rate': 'mean'
    }).round(2)
    
    # Monthly metrics
    monthly_metrics = merged_df.set_index('DATE').resample('ME').agg({
        'PPZFA': ['mean', 'median', 'count'],
        'PRICE': 'sum',
        'Rate': 'mean'
    }).round(2)
    
    # Calculate moving averages
    for window in [3, 6]:
        monthly_metrics[('PPZFA', f'MA{window}')] = monthly_metrics['PPZFA']['mean'].rolling(window=window).mean()
        monthly_metrics[('PRICE', f'MA{window}')] = monthly_metrics['PRICE']['sum'].rolling(window=window).mean()
    
    # Zoning analysis
    zoning_metrics = merged_df.groupby('ZONING 1').agg({
        'PPZFA': ['mean', 'count'],
        'PRICE': 'sum'
    }).round(2)
    
    # Rate period analysis
    rate_metrics = merged_df.groupby('Rate').agg({
        'PPZFA': ['mean', 'std', 'count'],
        'PRICE': ['sum', 'mean']
    }).round(2)
    
    return borough_metrics, monthly_metrics, zoning_metrics, rate_metrics

def calculate_correlations(monthly_metrics, lags=[1, 3, 6, 9]):
    """Calculate lagged correlations."""
    correlations = {}
    
    for lag in lags:
        correlations[f'Lag {lag} month'] = {
            'PPZFA vs Rate': monthly_metrics['PPZFA']['mean'].shift(lag).corr(monthly_metrics['Rate']['mean']),
            'Volume vs Rate': monthly_metrics['PRICE']['sum'].shift(lag).corr(monthly_metrics['Rate']['mean']),
            'Count vs Rate': monthly_metrics['PPZFA']['count'].shift(lag).corr(monthly_metrics['Rate']['mean'])
        }
    
    return correlations

def generate_report(merged_df, borough_metrics, monthly_metrics, zoning_metrics, rate_metrics, correlations):
    """Generate and print the analysis report."""
    print("\n=== MARKET ANALYSIS REPORT ===\n")
    
    # Overall metrics
    print("1. OVERALL MARKET METRICS")
    print("-----------------------")
    print(f"Total Transactions: {len(merged_df):,}")
    print(f"Total Volume: ${merged_df['PRICE'].sum():,.2f}")
    print(f"Average PPZFA: ${merged_df['PPZFA'].mean():.2f}")
    print(f"Analysis Period: {merged_df['DATE'].min().strftime('%m/%d/%Y')} to {merged_df['DATE'].max().strftime('%m/%d/%Y')}")
    
    # Rate impact
    print("\n2. FEDERAL RATE IMPACT")
    print("--------------------")
    for lag, metrics in correlations.items():
        print(f"\n{lag}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Borough analysis
    print("\n3. BOROUGH ANALYSIS")
    print("-----------------")
    print(borough_metrics)
    
    # Recent trends
    print("\n4. RECENT TRENDS (Last 6 months)")
    print("-------------------------------")
    print(monthly_metrics.tail(6))
    
    # Zoning analysis
    print("\n5. ZONING ANALYSIS")
    print("----------------")
    print("\nTop 5 Zones by Transaction Volume:")
    print(zoning_metrics.sort_values(('PRICE', 'sum'), ascending=False).head())
    
    # Rate period analysis
    print("\n6. RATE PERIOD ANALYSIS")
    print("---------------------")
    print(rate_metrics)

def main():
    """Main execution function."""
    # Load data
    fed_rates_df, transactions_df, output_dir = load_data()
    latest_transaction_date = transactions_df['DATE'].max()
    
    # Create expanded rate periods
    expanded_rates = expand_rate_periods(fed_rates_df, latest_transaction_date)
    
    # Merge datasets
    merged_df = pd.merge(transactions_df, expanded_rates, 
                        left_on='DATE', right_on='Date', how='left')
    
    # Calculate metrics
    borough_metrics, monthly_metrics, zoning_metrics, rate_metrics = calculate_metrics(merged_df)
    correlations = calculate_correlations(monthly_metrics)
    
    # Generate report
    generate_report(merged_df, borough_metrics, monthly_metrics, 
                   zoning_metrics, rate_metrics, correlations)
    
    # Save results
    monthly_metrics.to_csv(output_dir / 'monthly_analysis.csv')
    borough_metrics.to_csv(output_dir / 'borough_analysis.csv')
    rate_metrics.to_csv(output_dir / 'rate_analysis.csv')
    print("\nDetailed analysis has been saved to CSV files.")

if __name__ == "__main__":
    main()
