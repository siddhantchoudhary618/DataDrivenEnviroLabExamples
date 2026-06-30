import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import sys
import os

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Fit and plot emission or concentration data.')
parser.add_argument(
    'filename',
    nargs='?',
    default='mauna_loa_80yr_projection - mauna_loa_80yr_projection.csv.csv',
    help='Path to the emissions or concentration CSV file (default: co2_emissions.csv)'
)
parser.add_argument(
    '--single',
    action='store_true',
    help='Plot raw data, quadratic fit, and exponential fit on a single figure'
)
args = parser.parse_args()
filename = args.filename
single_plot = args.single

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
df.columns = df.columns.str.strip()

# ✅ Updated section to handle "Annual CO₂ emissions"
y_column = None
if 'Emission' in df.columns and 'Concentration' in df.columns:
    print("\n Error: Both 'Emission' and 'Concentration' columns found. Please ensure only one is present.")
    sys.exit(1)
elif 'Emission' in df.columns:
    y_column = 'Emission'
elif 'Concentration' in df.columns:
    y_column = 'Concentration'
elif 'Annual CO₂ emissions' in df.columns:
    y_column = 'Annual CO₂ emissions'
else:
    print("\n Error: No valid emissions column found. Expected 'Emission', 'Concentration', or 'Annual CO₂ emissions'.")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

# --- Clean data ---
df['Entity'] = df['Entity'].str.strip()
df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
df.dropna(subset=[y_column], inplace=True)

entities = df['Entity'].unique()
palette = sns.color_palette("husl", len(entities))
color_map = dict(zip(entities, palette))

# --- Define model functions ---
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def expo(x, A, B):
    return A * np.exp(B * x)

min_year = df['Year'].min()

# --- Check if "Plots" directory exists, if not create it ---
plots_dir = 'Plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# --- Function to Plot Data ---
def plot_data(entities, single_plot=False):
    if single_plot:
        plt.figure(figsize=(14, 8))
    
    for i, entity in enumerate(entities):
        subset = df[df['Entity'] == entity].sort_values('Year')
        x = subset['Year'].values
        y = subset[y_column].values
        t = x - min_year  # normalized year for fitting

        if len(x) < 5:
            continue

        # --- Polynomial Fit (Quadratic) ---
        try:
            popt_poly, _ = curve_fit(poly2, t, y, maxfev=2000)
            a, b, c = popt_poly
            y_poly_pred = poly2(t, a, b, c)
            if single_plot:
                plt.plot(x, y_poly_pred, label=f'{entity} - Quadratic Fit', color=color_map[entity], linestyle='-', linewidth=2)
            print(f"\n{entity} - Quadratic Fit:")
            print(f"a = {a:.4e}")
            print(f"b = {b:.4e}")
            print(f"c = {c:.4e}")
        except Exception as e:
            print(f"Polynomial fit error for {entity}: {e}")
        
        # --- Exponential Fit ---
        try:
            mask = y > 0
            x_exp = x[mask]
            y_exp = y[mask]
            t_exp = x_exp - min_year  # normalized year for fitting
            popt_exp, _ = curve_fit(expo, t_exp, y_exp, p0=(1, 0.01), maxfev=2000)
            A, B = popt_exp
            y_exp_pred = expo(t_exp, A, B)
            if single_plot:
                plt.plot(x_exp, y_exp_pred, label=f'{entity} - Exponential Fit', color=color_map[entity], linestyle='--', linewidth=2)
            print(f"\n{entity} - Exponential Fit:")
            print(f"A = {A:.4e}")
            print(f"B = {B:.4e}")
        except Exception as e:
            print(f"Exponential fit error for {entity}: {e}")

        # --- Raw Data ---
        if single_plot:
            plt.scatter(x, y, s=30, label=f'{entity} - Raw Data', color=color_map[entity], alpha=0.7, edgecolor='black')

    if single_plot:
        plt.xlabel('Year')
        plt.ylabel(y_column.capitalize())
        plt.title(f'{y_column.capitalize()} Data, Quadratic and Exponential Fits by Entity')
        plt.legend(fontsize='small', title='Entity', loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, f'{y_column.upper()}_ALL_curvefit.pdf'), bbox_inches='tight')
        plt.show()
    else:
        plot_separate(entities)

def plot_separate(entities):
    # Plot Raw Data and Quadratic Fit
    plt.figure(figsize=(14, 8))
    for i, entity in enumerate(entities):
        subset = df[df['Entity'] == entity].sort_values('Year')
        x = subset['Year'].values
        y = subset[y_column].values
        t = x - min_year

        if len(x) < 5:
            continue

        try:
            popt_poly, _ = curve_fit(poly2, t, y, maxfev=2000)
            a, b, c = popt_poly
            y_poly_pred = poly2(t, a, b, c)
            plt.plot(x, y_poly_pred, label=f'{entity} - Quadratic Fit', color=color_map[entity], linestyle='-', linewidth=2)
            print(f"\n{entity} - Quadratic Fit:")
            print(f"a = {a:.4e}")
            print(f"b = {b:.4e}")
            print(f"c = {c:.4e}")
        except Exception as e:
            print(f"Polynomial fit error for {entity}: {e}")
        
        plt.scatter(x, y, s=30, label=f'{entity} - Raw Data', color=color_map[entity], alpha=0.7, edgecolor='black')

    plt.xlabel('Year')
    plt.ylabel(y_column.capitalize())
    plt.title(f'{y_column.capitalize()} Data and Quadratic Fits by Entity')
    plt.legend(fontsize='small', title='Entity', loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{y_column.upper()}_Raw_and_Quadratic.pdf'), bbox_inches='tight')
    plt.show()

    # Plot Raw Data and Exponential Fit
    plt.figure(figsize=(14, 8))
    for i, entity in enumerate(entities):
        subset = df[df['Entity'] == entity].sort_values('Year')
        x = subset['Year'].values
        y = subset[y_column].values

        mask = y > 0
        x_exp = x[mask]
        y_exp = y[mask]
        t_exp = x_exp - min_year

        if len(x_exp) < 5:
            continue

        try:
            popt_exp, _ = curve_fit(expo, t_exp, y_exp, p0=(1, 0.01), maxfev=2000)
            A, B = popt_exp
            y_exp_pred = expo(t_exp, A, B)
            plt.plot(x_exp, y_exp_pred, label=f'{entity} - Exponential Fit', color=color_map[entity], linestyle='--', linewidth=2)
            print(f"\n{entity} - Exponential Fit:")
            print(f"A = {A:.4e}")
            print(f"B = {B:.4e}")
        except Exception as e:
            print(f"Exponential fit error for {entity}: {e}")

        plt.scatter(x, y, s=30, label=f'{entity} - Raw Data', color=color_map[entity], alpha=0.7, edgecolor='black')

    plt.xlabel('Year')
    plt.ylabel(y_column.capitalize())
    plt.title(f'{y_column.capitalize()} Data and Exponential Fits by Entity')
    plt.legend(fontsize='small', title='Entity', loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{y_column.upper()}_Raw_and_Exponential.pdf'), bbox_inches='tight')
    plt.show()

# --- Main Execution ---
plot_data(entities, single_plot=single_plot)
