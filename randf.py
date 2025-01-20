import pandas as pd
import statsmodels.api as sm

# Load the dataset
file_path = 'Advertising.csv'
advertising_data = pd.read_csv(file_path)

# Prepare the independent variables (TV, radio, newspaper) and dependent variable (sales)
X = advertising_data[['TV', 'radio', 'newspaper']]
y = advertising_data['sales']

# Add a constant to the independent variables for the intercept term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Calculate the Residual Standard Error (RSE)
rse = model.mse_resid ** 0.5  # Residual Standard Error is the square root of MSE of residuals

# Calculate R-squared
r_squared = model.rsquared  # Coefficient of determination

# Calculate F-statistic
f_statistic = model.fvalue  # F-statistic for overall model significance

# Print the results
print(f"Residual Standard Error (RSE): {rse}")
print(f"R-squared: {r_squared}")
print(f"F-statistic: {f_statistic}")
