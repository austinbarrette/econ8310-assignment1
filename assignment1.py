"""
Econ 8310 – Assignment 1
Business Forecasting

This script uses the Exponential Smoothing model
to forecast hourly NYC taxi trip demand for January of the following year.

The approach mirrors the methodology presented in Lesson 1:
- Begin with exponential smoothing concepts
- Extend to include trend
- Further extend to include seasonality
"""

# Imports numpy, pandas and the exponential smoothing model
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

# Load training and test data
train = pd.read_csv(
    "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
)
test = pd.read_csv(
    "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
)

# Clean and Sort the data
# Convert timestamps to datetime to ensure correct ordering
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
test["Timestamp"] = pd.to_datetime(test["Timestamp"])

# Sort by time to preserve the time-series structure
train = train.sort_values("Timestamp")
test = test.sort_values("Timestamp")

# Dependent variable = trips
y = train.set_index("Timestamp")["trips"].astype(float).asfreq("h")


# Model definition (Lesson 1)
"""
Exponential Smoothing models generate forecasts using weighted averages of past observations,
where recent observations receive more weight than older ones.

Simple exponential smoothing produces flat forecasts and assumes a constant level.
What we have to look for with hourly taxi demand are:
- A changing level
- Local trends
- Strong recurring weekly seasonality

To address this, we use the Holt-Winters method (triple exponential smoothing),
which explicitly models:
- Level
- Trend
- Seasonality
"""

# Define the Holt-Winters Exponential Smoothing model
# Seasonal period = 168 hours (24 hours × 7 days) to capture weekly seasonality
#model = ExponentialSmoothing(
#    y,
#    trend="add",
#    seasonal="add",
#    seasonal_periods=168
#)

model = ExponentialSmoothing(
    y,
    trend="add",
    damped_trend=True,
    seasonal="mul",
    seasonal_periods=168,
    initialization_method="estimated"
)


# Fit the model
"""
Rather than manually selecting smoothing parameters, we allow the model
to optimize them based on the observed data. This determines how quickly
the model adapts to recent changes while retaining information from the past.
"""

#modelFit = model.fit(optimized=True)
modelFit = model.fit(optimized=True, use_boxcox=True, use_brute=True)


# Generate forecasts
"""
Once fitted, the Holt-Winters model extrapolates the estimated level and trend
while repeating the learned weekly seasonal pattern into the future.

The assignment requires forecasts for each hour in January of the following year,
which corresponds to 744 total hours.
"""

#pred = modelFit.forecast(744)
pred = np.asarray(modelFit.forecast(744))

# Convert to a NumPy array to ensure compatibility with grading tests
pred = np.asarray(pred)
