from nat_gas_prices import *
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from datetime import datetime
from scipy.optimize import curve_fit

class GasStorageContract:
    def __init__(self, historical_data_path: str = 'Nat_Gas.csv'):
        """
        Initialize the pricing model with historical data and fit the price model.
        
        Args:
            historical_data_path: Path to CSV file with historical price data
        """
        # Load and prepare historical data
        self.df = pd.read_csv(historical_data_path, parse_dates=['Dates']).sort_values('Dates')
        
        # Convert dates to ordinal for modeling
        self.df['DateOrdinal'] = self.df['Dates'].apply(lambda date: date.toordinal())
        
        # Fit the price model
        self.params = self._fit_price_model()
        
        # Store the last historical date
        self.last_historical_date = self.df['Dates'].max()
    
    def _fit_price_model(self):
        """Fit the seasonal price model to historical data."""
        def price_model(t, a, b, c, d, e):
            """Mathematical model for natural gas price."""
            return a * t + b * np.sin(2 * np.pi * (t - e) / 365.25) + c * np.cos(2 * np.pi * (t - e) / 365.25) + d
        
        t_data = self.df['DateOrdinal'].values
        price_data = self.df['Prices'].values
        
        initial_guesses = [0.001, 2, 2, 10, 0]
        params, _ = curve_fit(price_model, t_data, price_data, p0=initial_guesses)
        return params
    
    def estimate_price(self, date: Union[str, datetime, pd.Timestamp]) -> float:
        """
        Estimate the gas price for a specific date using the fitted model.
        
        Args:
            date: Date to estimate price for (YYYY-MM-DD format, datetime, or Timestamp)
            
        Returns:
            Estimated price for the given date
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        elif isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        
        ordinal_date = date.toordinal()
        
        # Use the fitted model parameters to estimate price
        a, b, c, d, e = self.params
        estimated_price = (a * ordinal_date + 
                          b * np.sin(2 * np.pi * (ordinal_date - e) / 365.25) + 
                          c * np.cos(2 * np.pi * (ordinal_date - e) / 365.25) + d)
        
        # For dates beyond historical data, apply a small uncertainty buffer
        if date > self.last_historical_date:
            # Simple approach: add 5% buffer for future uncertainty
            return estimated_price * 1.05
        return estimated_price
    
    def price_contract(
        self,
        injection_dates: List[Union[str, datetime, pd.Timestamp]],
        withdrawal_dates: List[Union[str, datetime, pd.Timestamp]],
        injection_rates: List[float],
        withdrawal_rates: List[float],
        max_storage: float,
        storage_cost: float,
        injection_volume: float = None,
        withdrawal_volume: float = None,
        discount_rate: float = 0.0  # Now optional for future extensibility
    ) -> Dict:
        """
        Price a natural gas storage contract with enhanced price estimation.
        
        Args:
            injection_dates: List of dates when gas is injected
            withdrawal_dates: List of dates when gas is withdrawn
            injection_rates: Rate of injection (volume per day) for each injection date
            withdrawal_rates: Rate of withdrawal (volume per day) for each withdrawal date
            max_storage: Maximum storage capacity
            storage_cost: Daily cost per unit of gas stored
            injection_volume: Optional fixed total injection volume (overrides rates)
            withdrawal_volume: Optional fixed total withdrawal volume (overrides rates)
            discount_rate: Optional daily discount rate for NPV calculation
            
        Returns:
            Dictionary containing:
                - total_value: Net present value of the contract
                - cash_flows: DataFrame of all cash flows
                - storage_profile: DataFrame of storage levels over time
                - summary: Dictionary with key metrics
                - price_estimates: DataFrame of price estimates used
        """
        # Convert all dates to datetime objects
        injection_dates = [self._parse_date(d) for d in injection_dates]
        withdrawal_dates = [self._parse_date(d) for d in withdrawal_dates]
        
        # Sort all dates chronologically
        all_dates = sorted(injection_dates + withdrawal_dates)
        
        # Initialize storage tracking
        storage_level = 0.0
        current_day = 0
        
        # Prepare results storage
        cash_flows = []
        storage_levels = []
        price_estimates = []
        
        # Calculate total volumes if fixed amounts are provided
        if injection_volume is not None:
            total_injection_days = len(injection_dates)
            injection_rates = [injection_volume / total_injection_days] * len(injection_dates)
            
        if withdrawal_volume is not None:
            total_withdrawal_days = len(withdrawal_dates)
            withdrawal_rates = [withdrawal_volume / total_withdrawal_days] * len(withdrawal_dates)
        
        # Process each date in order
        for date in all_dates:
            daily_storage_cost = 0.0
            daily_injection = 0.0
            daily_withdrawal = 0.0
            
            # Get price estimate for this date
            price = self.estimate_price(date)
            price_estimates.append({'date': date, 'price': price})
            
            # Check if this is an injection date
            if date in injection_dates:
                rate = injection_rates[injection_dates.index(date)]
                daily_injection = min(rate, max_storage - storage_level)  # Can't exceed available capacity
                
                cash_flows.append({
                    'date': date,
                    'type': 'injection',
                    'volume': daily_injection,
                    'price': price,
                    'cost': -daily_injection * price,
                    'storage_cost': 0,
                    'discount_factor': 1 / ((1 + discount_rate) ** current_day)
                })
            
            # Check if this is a withdrawal date
            if date in withdrawal_dates:
                rate = withdrawal_rates[withdrawal_dates.index(date)]
                daily_withdrawal = min(rate, storage_level)  # Can't withdraw more than available
                
                cash_flows.append({
                    'date': date,
                    'type': 'withdrawal',
                    'volume': daily_withdrawal,
                    'price': price,
                    'cost': daily_withdrawal * price,
                    'storage_cost': 0,
                    'discount_factor': 1 / ((1 + discount_rate) ** current_day)
                })
            
            # Update storage level
            storage_level += daily_injection - daily_withdrawal
            
            # Calculate storage costs for the day
            daily_storage_cost = storage_level * storage_cost
            if daily_storage_cost > 0:
                cash_flows.append({
                    'date': date,
                    'type': 'storage',
                    'volume': storage_level,
                    'price': storage_cost,
                    'cost': -daily_storage_cost,
                    'storage_cost': daily_storage_cost,
                    'discount_factor': 1 / ((1 + discount_rate) ** current_day)
                })
            
            storage_levels.append({
                'date': date,
                'level': storage_level,
                'injected': daily_injection,
                'withdrawn': daily_withdrawal
            })
            
            current_day += 1
        
        # Convert results to DataFrames
        cash_flows_df = pd.DataFrame(cash_flows)
        storage_profile_df = pd.DataFrame(storage_levels)
        price_estimates_df = pd.DataFrame(price_estimates)
        
        # Calculate total value (NPV)
        cash_flows_df['discounted_cost'] = cash_flows_df['cost'] * cash_flows_df['discount_factor']
        total_value = cash_flows_df['discounted_cost'].sum()
        
        # Prepare summary
        summary = {
            'total_value': total_value,
            'total_injection': cash_flows_df[cash_flows_df['type'] == 'injection']['volume'].sum(),
            'total_withdrawal': cash_flows_df[cash_flows_df['type'] == 'withdrawal']['volume'].sum(),
            'total_storage_cost': cash_flows_df['storage_cost'].sum(),
            'peak_storage': storage_profile_df['level'].max(),
            'average_storage': storage_profile_df['level'].mean(),
            'injection_cost': cash_flows_df[cash_flows_df['type'] == 'injection']['cost'].sum(),
            'withdrawal_revenue': cash_flows_df[cash_flows_df['type'] == 'withdrawal']['cost'].sum(),
            'net_storage_cost': cash_flows_df[cash_flows_df['type'] == 'storage']['cost'].sum()
        }
        
        return {
            'total_value': total_value,
            'cash_flows': cash_flows_df,
            'storage_profile': storage_profile_df,
            'price_estimates': price_estimates_df,
            'summary': summary
        }
    
    def _parse_date(self, date: Union[str, datetime, pd.Timestamp]) -> datetime:
        """Helper method to parse different date formats."""
        if isinstance(date, str):
            return datetime.strptime(date, '%Y-%m-%d')
        elif isinstance(date, pd.Timestamp):
            return date.to_pydatetime()
        return date

    def generate_price_curve(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate a price curve between two dates.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with dates and estimated prices
        """
        dates = pd.date_range(start=start_date, end=end_date)
        prices = [self.estimate_price(date) for date in dates]
        return pd.DataFrame({'Date': dates, 'Price': prices})

pricing_model = GasStorageContract('Nat_Gas.csv')

injection_dates = []
withdrawal_dates = []
injection_rates = []  # units per day
withdrawal_rates = []  # units per day
max_storage =   # units
storage_cost =   # per unit per day

# Price the contract
contract_result = pricing_model.price_contract(
    injection_dates,
    withdrawal_dates,
    injection_rates,
    withdrawal_rates,
    max_storage,
    storage_cost
)

# Display results
print(f"Contract NPV: ${contract_result['total_value']:,.2f}")
print("\nSummary Metrics:")
for k, v in contract_result['summary'].items():
    print(f"{k.replace('_', ' ').title()}: {v:,.2f}")

# Save results to CSV
contract_result['cash_flows'].to_csv('contract_cash_flows.csv', index=False)
contract_result['storage_profile'].to_csv('contract_storage_profile.csv', index=False)