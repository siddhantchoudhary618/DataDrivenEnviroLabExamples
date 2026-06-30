import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Loading the data and filtering it
def load_data(filename, regions):
    data = pd.read_csv(filename)
    #Renaming columns so they are easier to type later
    data = data.rename(columns={'Entity': 'Region', 'Annual CO₂ emissions': 'Emissions'})
    
    # I decided to just look at data from 2007 since that was when I was born
    clean_df = data[(data['Year'] >= 2007) & (data['Region'].isin(regions))]
    return clean_df

#Making predictions for the data based on a linear model
def make_predictions(data):
    results = []
    regions = data['Region'].unique()
    
    for i in regions:
        country_data = data[data['Region'] == i]
        x = country_data[['Year']].values
        y = country_data['Emissions'].values
        
        #Linear Model
        model = LinearRegression()
        model.fit(x, y)
        
        #Predicting up to 2051. I started at 2025 since the data ends at 2024
        for year in range(2025, 2051):
            predict = model.predict([[year]])[0]
            #Making sure we don't show negative emissions
            if predict < 0:
                predict = 0
            results.append({'Region': i, 'Year': year, 'Emissions': predict})
            
    return pd.DataFrame(results)

#Setting the file path
file = 'annual-co2-emissions-per-country.csv'
countries = ['United States', 'Africa', 'China', 'South America', 'North America']

#Running functions, hd is short for historical data
hd = load_data(file, countries)
projections = make_predictions(hd)

#Creating the plot to show history and future projections
plt.figure()
for i in countries:
    # Displaying historical data
    subset = hd[hd['Region'] == i]
    line, = plt.plot(subset['Year'], subset['Emissions'], label=i)

    linecolor = line.get_color()
    
    #Drawing projected trends
    proj_subset = projections[projections['Region'] == i]
    plt.plot(proj_subset['Year'], proj_subset['Emissions'], linestyle='dashdot', color = linecolor)

plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2 Emissions to the power of 10 (tonnes)")
plt.title("Historical and Predicted CO2 Trends by Region")
plt.show()