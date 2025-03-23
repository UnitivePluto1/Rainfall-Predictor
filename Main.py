# IMPORT ALL THE NECESSARY LIBRARIES
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression  # Our ML Regression model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Splitting of data

'''
# Initial Steps


## Read the data
data = pd.read_csv("Datasets/austin_weather.csv")

## Deleting useless features
data = data.drop(['Date','Events','SeaLevelPressureHighInches','SeaLevelPressureLowInches'], axis=1)

## Replace null and T values
data = data.replace('T', 0.0)
data = data.replace('-',0.0)


## Convert to a final dataset
data.to_csv("Datasets/weather_final.csv")
'''
# 
## Read final data
data = pd.read_csv("Datasets/weather_final.csv")

## Defining our Features and Labels
X = data.drop("PrecipitationSumInches",axis = 1)        ### Features
y = data["PrecipitationSumInches"]                      ### Label
y = y.values.reshape(-1, 1)                             ### Converting it into a 2d vector for the ML model to read. 
                                                        ### Might cause errors if it were a 2d array
## Instantiate model and train it
clf = LinearRegression()
clf.fit(X, y)

## Dayindex and Daynumber (The Precipitation of which is to be predicted)
day_index = 405
days = [i for i in range(y.size)]

## Sample Input
val1 = np.array([[405], [58], [53], [48], [39], [36], [33], [63], [52], [40], [30.37], [10], [10], [10], [14], [6], [23]])
val1 = val1.reshape(1, -1)

## Prediction
print("\nThe precipitation in inches for the given value is\t--\t",clf.predict(val1))

## Plot for values of PrecipitationSumInches column on different days
print("\nThe precipitation trend graph: ")
fig, ax = plt.subplots()                                ### Object Oriented method of plotting
scatter = ax.scatter(
        x = days,
        y = y,
        c = 'g')                                        ### Values will be displayed as a green coloured dot
scatter_main = ax.scatter(
        x = days[day_index],
        y = y[day_index],
        c = 'r')                                        ### The days[day_index]'s value will be displayed as a red coloured dot
ax.set_title("Precipitation Levels")
ax.set_xlabel("Days")
ax.set_ylabel("Precipitation Inches")

## Display final plot
plt.show()

## Plots for Precipitation vs other selected attributes
x_f = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
                'WindAvgMPH'], axis=1)
print('Preciptiation Vs Selected Attributes Graph: ')
for i in range(x_f.columns.size):
    plt.subplot(3, 2, i+1)
    plt.scatter(days, x_f[x_f.columns.values[i][:100]], color='g')
    plt.scatter(days[day_index], x_f[x_f.columns.values[i]]
                [day_index], color='r')
    plt.title(x_f.columns.values[i])

## Display the final graph with a few features vs precipitation to observe the trends
plt.show()