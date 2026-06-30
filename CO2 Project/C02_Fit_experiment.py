import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import seaborn as sns
import sys

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Fit and plot emission data.')
parser.add_argument(
    'filename',
    nargs='?',
    default='co2_emissions.csv',
    help='/Users/siddhantchoudhary/Downloads/CO2_project/co2_emissions_world.csv'
)
args = parser.parse_args()
filename = args.filename

# --- Load and validate CSV file ---
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"\n Error: File '{filename}' not found. Please check the path and try again.")
    sys.exit(1)
except Exception as e:
    print(f"\n Error reading file '{filename}': {e}")
    sys.exit(1)

# --- Validate required columns ---
required_columns = {'Entity', 'Emission'}  # Updated column name
df.columns = df.columns.str.strip()  # Strip any surrounding spaces
missing_columns = required_columns - set(df.columns)

if missing_columns:
    print(f"\n Error: The following required column(s) are missing from the CSV: {', '.join(missing_columns)}")
    sys.exit(1)

# --- Clean data ---
df['Entity'] = df['Entity'].str.strip()  # Strip any surrounding spaces
df['Emission'] = pd.to_numeric(df['Emission'], errors='coerce')  # Changed to 'Emission'
df.dropna(subset=['Emission'], inplace=True)

entities = df['Entity'].unique()

if entities.size > 1:
    print(f"\n Error: Too many entities in the CSV: {', '.join(entities)}")
    sys.exit(1)

palette = sns.color_palette("husl", len(entities))
color_map = dict(zip(entities, palette))

# Prepare variables
x = df['Year'].values
y = df['Emission'].values  # Changed to 'Emission'

# Normalize inputs
min_year = x.min()
t = x - min_year  # Normalize year
y_max = y.max()
y_norm = y / y_max  # Normalize emissions

# Define models
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, A, B):
    return A * np.exp(B * x)  # Regular exponential A * exp(B * x)

# --- Linear fit ---
popt_lin, _ = curve_fit(linear, t, y_norm)
a_lin, b_lin = popt_lin * y_max
y_lin_norm = linear(t, *popt_lin)
mse_lin_norm = mean_squared_error(y_norm, y_lin_norm)

# --- Quadratic fit ---
popt_quad, _ = curve_fit(quadratic, t, y_norm, maxfev=10000)
a_quad, b_quad, c_quad = popt_quad * y_max
y_quad_norm = quadratic(t, *popt_quad)
mse_quad_norm = mean_squared_error(y_norm, y_quad_norm)

# --- Exponential fit ---
initial_A = y.max()  # Start with the max emission
initial_B = 1e-8  # A small positive B for growth

# Fit the regular exponential model A * exp(B * x)
popt_exp, _ = curve_fit(exponential, t, y_norm, p0=(initial_A, initial_B), maxfev=10000)

A_exp_rescaled = popt_exp[0] * y_max
B_exp = popt_exp[1]
y_exp_norm_fit = exponential(t, *popt_exp)

# Calculate MSE for exponential model
mse_exp_norm = mean_squared_error(y_norm, y_exp_norm_fit)

# --- Print the results with formatted output ---
print(f"\nLinear fit over years (t): from t=0 ({int(min_year)}) to t={int(t[-1])} ({int(x[-1])})")
print(f"  Linear Fit Parameters (rescaled): a = {a_lin: .4e}, b = {b_lin: .4e}")
print(f"  Linear MSE (normalized): {mse_lin_norm: .4e}")

print(f"\nQuadratic fit over years (t): from t=0 ({int(min_year)}) to t={int(t[-1])} ({int(x[-1])})")
print(f"  Quadratic Fit Parameters (rescaled): a = {a_quad: .4e}, b = {b_quad: .4e}, c = {c_quad: .4e}")
print(f"  Quadratic MSE (normalized): {mse_quad_norm: .4e}")

print(f"\nExponential fit over years (t): from t=0 ({int(min_year)}) to t={int(t[-1])} ({int(x[-1])})")
print(f"  Exponential Fit Parameters (rescaled): A = {A_exp_rescaled: .4e}, B = {B_exp: .4e}")
print(f"  Exponential MSE (normalized): {mse_exp_norm: .4e}")

# --- Plot all fits ---
plt.figure(figsize=(12, 7))

# Scatter plot of the raw data
plt.scatter(x, y, color='black', s=20, alpha=0.5, label='Observed Emissions')

# Plot Linear Fit
plt.plot(x, y_lin_norm * y_max, label='Linear Fit')
# Plot Quadratic Fit
plt.plot(x, y_quad_norm * y_max, label='Quadratic Fit')
# Plot Exponential Fit
plt.plot(x, y_exp_norm_fit * y_max, label='Exponential Fit')

# Finalize plot
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions')
plt.title('CO₂ Emissions Trend Fits from '+str(int(min_year)) + ' to ' + str(int(x[-1]) ))

# Set x-axis ticks to be only integers (Years)
# year_ticks = np.arange(int(x.min()), int(x.max()) + 1, 5)  # Every 5 years
# plt.xticks(year_ticks)

# Add legend
plt.legend()

# Add grid
plt.grid(True)

# Adjust layout to avoid clipping of labels
plt.tight_layout()

# Show plot
plt.show()
