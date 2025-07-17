import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta

# --- 1. Load and Prepare Data ---

# Load the dataset from the CSV file
# The `parse_dates` argument tells pandas to interpret the 'Dates' column as dates
try:
    df = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'])
except FileNotFoundError:
    print("Error: 'Nat_Gas.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Sort the DataFrame by dates to ensure chronological order
df = df.sort_values('Dates')

# --- 2. Initial Data Visualization ---

# Create a plot of the original monthly price data
plt.figure(figsize=(12, 6))
plt.plot(df['Dates'], df['Prices'], 'o-', label='Historical Monthly Prices')
plt.title('Natural Gas Prices (End of Month)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
# Save the plot to a file
plt.savefig('natural_gas_prices.png')
print("Saved initial price plot to 'natural_gas_prices.png'")
plt.close()

# --- 3. Modeling ---

# Convert dates to a numerical format (ordinal days) for mathematical modeling
# Ordinal days represent the number of days since a fixed point in time (day 1 of year 1)
df['DateOrdinal'] = df['Dates'].apply(lambda date: date.toordinal())

# Define the model function.
# This function combines a linear component (a*t + d) for the trend
# and two sinusoidal components (sin and cos) for the annual seasonality.
# t: time in ordinal days
# a: slope, determines the rate of the linear trend
# b, c: amplitudes of the sine and cosine waves, determine the magnitude of seasonal swings
# d: y-intercept, the baseline price at the start
# e: phase shift, adjusts the seasonal cycle to fit the data's peaks and troughs
def price_model(t, a, b, c, d, e):
    """Mathematical model for natural gas price."""
    return a * t + b * np.sin(2 * np.pi * (t - e) / 365.25) + c * np.cos(2 * np.pi * (t - e) / 365.25) + d

# --- 4. Curve Fitting ---

# Extract the data for fitting the model
t_data = df['DateOrdinal'].values
price_data = df['Prices'].values

# Use `curve_fit` from the scipy library to find the optimal parameters (a, b, c, d, e)
# that make the `price_model` best fit the historical data.
# `p0` provides initial guesses for the parameters to help the algorithm converge.
initial_guesses = [0.001, 2, 2, 10, 0]
params, covariance = curve_fit(price_model, t_data, price_data, p0=initial_guesses)

# --- 5. Extrapolation and Visualization of Fitted Model ---

# Generate a continuous range of dates for plotting the fitted curve
fit_dates = pd.to_datetime(pd.date_range(start=df['Dates'].min(), end=df['Dates'].max()))
fit_ordinals = np.array([date.toordinal() for date in fit_dates])

# Generate future dates for the one-year extrapolation
last_date = df['Dates'].max()
extrapolation_dates = pd.to_datetime([last_date + timedelta(days=x) for x in range(1, 366)])
extrapolation_ordinals = np.array([date.toordinal() for date in extrapolation_dates])

# Calculate the prices on the fitted curve and for the extrapolated period using the optimized parameters
fitted_prices = price_model(fit_ordinals, *params)
extrapolated_prices = price_model(extrapolation_ordinals, *params)

# Plot the final results: historical data, the fitted model, and the extrapolation
plt.figure(figsize=(14, 7))
plt.plot(df['Dates'], df['Prices'], 'o', label='Historical Monthly Prices')
plt.plot(fit_dates, fitted_prices, '-', label='Fitted Price Curve')
plt.plot(extrapolation_dates, extrapolated_prices, '--', label='Extrapolated Prices')
plt.title('Natural Gas Prices: Historical, Fitted, and Extrapolated')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
# Save the final plot to a file
plt.savefig('natural_gas_price_forecast.png')
print("Saved forecast plot to 'natural_gas_price_forecast.png'")
plt.close()

# --- 6. Save Extrapolated Data ---

# Create a DataFrame for the extrapolated data and save it to a CSV file
extrapolated_df = pd.DataFrame({'Dates': extrapolation_dates, 'Prices': extrapolated_prices})
extrapolated_df.to_csv('extrapolated_natural_gas_prices.csv', index=False)
print("Saved extrapolated prices to 'extrapolated_natural_gas_prices.csv'")


# --- 7. Price Estimation Function ---

def estimate_price(date_str):
    """
    Estimates the price of natural gas for a given date using the fitted model.
    :param date_str: Date in 'YYYY-MM-DD' format.
    :return: Estimated price as a formatted string or an error message.
    """
    try:
        # Convert the input date string to a datetime object
        input_date = datetime.strptime(date_str, '%Y-%m-%d')
        # Convert the datetime object to an ordinal number
        ordinal_date = input_date.toordinal()
        # Calculate the price using the model and the optimized parameters
        estimated_price = price_model(ordinal_date, *params)
        return f"${estimated_price:.2f}"
    except ValueError:
        return "Invalid date format. Please use 'YYYY-MM-DD'."

# --- Example Usage of the Function ---
user_date = input("Enter data in yyyy-mm-dd format")
print(f"Estimated price for {user_date}: {estimate_price(user_date)}")
