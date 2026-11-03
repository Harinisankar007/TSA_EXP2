# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### DATASET 
DailyDelhiClimate

### SOFTWARE REQUIRED 
Jupyter Notebook

### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the training and testing datasets
train = pd.read_csv(
    "DailyDelhiClimateTrain.csv",
    parse_dates=["date"]
)

test = pd.read_csv(
    "DailyDelhiClimateTest.csv",
    parse_dates=["date"]
)

# 2. Clean and sort the data
target = "meantemp"

for data in (train, test):
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data.dropna(subset=["date", target], inplace=True)
    data.sort_values("date", inplace=True)
    data.drop_duplicates(subset="date", inplace=True)
    data.reset_index(drop=True, inplace=True)

if train.empty or test.empty:
    raise ValueError("Both files must contain valid date and meantemp values.")

# Remove any shared dates from training so test dates remain unseen
train = train.loc[~train["date"].isin(test["date"])].copy()

if train["date"].nunique() < 3:
    raise ValueError("At least 3 distinct training dates are required.")

print("Training Data:")
print(train.head().to_string(index=False))

print("\nTesting Data:")
print(test.head().to_string(index=False))

# 3. Create the time variable
# x = number of days from the middle training date
reference_date = train["date"].iloc[len(train) // 2]

X_train = (
    train["date"] - reference_date
).dt.days.to_numpy(dtype=float)

X_test = (
    test["date"] - reference_date
).dt.days.to_numpy(dtype=float)

y_train = train[target].to_numpy(dtype=float)
y_test = test[target].to_numpy(dtype=float)

```
#### A - LINEAR TREND ESTIMATION
```python
# 4. Linear trend estimation: y = a + bx
n = len(X_train)

sum_x = np.sum(X_train)
sum_y = np.sum(y_train)
sum_x2 = np.sum(X_train ** 2)
sum_xy = np.sum(X_train * y_train)

b = (  n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
a = (sum_y - b * sum_x) / n
linear_train = a + b * X_train
linear_test = a + b * X_test
```


#### B- POLYNOMIAL TREND ESTIMATION
```python
# 5. Polynomial trend estimation (degree 2): y = a + bx + cx²
# Least-squares solution using the columns 1, x, and x²
design_train = np.column_stack([np.ones(n), X_train, X_train ** 2])
a_poly, b_poly, c_poly = np.linalg.lstsq(design_train,y_train, rcond=None)[0]

poly_train = ( a_poly+ b_poly * X_train + c_poly * X_train ** 2)

poly_test = ( a_poly+ b_poly * X_test+ c_poly * X_test ** 2)

```

```python
# 6. Display trend equations
print("\nTREND EQUATIONS")
print(f"x = number of days from {reference_date:%Y-%m-%d}")
print( f"\nLinear Trend: "f"y = {a:.6f} {b:+.6f}x")
print( f"\nPolynomial Trend: " f"y = {a_poly:.6f} " f"{b_poly:+.6f}x "f"{c_poly:+.8f}x²")

# 7. Store fitted values and test predictions
train["Linear Trend"] = linear_train
train["Polynomial Trend"] = poly_train

test["Linear Trend"] = linear_test
test["Polynomial Trend"] = poly_test

print("\nTEST PREDICTIONS")
print(  test[["date", target, "Linear Trend", "Polynomial Trend"] ].head(10).to_string(index=False))

# 8. Evaluate predictions on the test dataset
linear_mae = np.mean(np.abs(y_test - linear_test))
poly_mae = np.mean(np.abs(y_test - poly_test))

linear_rmse = np.sqrt(np.mean((y_test - linear_test) ** 2))
poly_rmse = np.sqrt(np.mean((y_test - poly_test) ** 2))

metrics = pd.DataFrame({
    "Model": ["Linear Trend", "Polynomial Trend (Degree 2)"],
    "Test MAE": [linear_mae, poly_mae],
    "Test RMSE": [linear_rmse, poly_rmse]
})

print("\nTEST EVALUATION")
print(metrics.round(4).to_string(index=False))

# Combine dates only for drawing smooth trend lines
all_dates = pd.DatetimeIndex( pd.concat([train["date"], test["date"]]).drop_duplicates(.sort_values())
X_all = (all_dates - reference_date).days.to_numpy(dtype=float)

linear_all = a + b * X_all
poly_all = a_poly + b_poly * X_all + c_poly * X_all ** 2

# 9. Linear trend estimation plot
plt.figure(figsize=(12, 5))

plt.plot(train["date"], y_train,  color="blue", linewidth=1, alpha=0.7,label="Actual Training Temperature")

plt.plot(test["date"], y_test, color="green", linewidth=1.5, label="Actual Test Temperature")

plt.plot( all_dates, linear_all,color="black", linestyle="--", linewidth=2, label="Linear Trend / Test Prediction")

plt.axvline(test["date"].min(),color="red", linestyle=":", label="Test Start")

plt.title("Linear Trend Estimation - Delhi Climate")
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 10. Polynomial trend estimation plot
plt.figure(figsize=(12, 5))
plt.plot( train["date"], y_train,color="blue", linewidth=1, alpha=0.7,label="Actual Training Temperature")
plt.plot(test["date"], y_test,color="green", linewidth=1.5,label="Actual Test Temperature")
plt.plot(all_dates, poly_all,color="black", linestyle="--", linewidth=2,label="Polynomial Trend/Test Prediction")
plt.axvline(test["date"].min(), color="red", linestyle=":", label="Test Start")
plt.title("Polynomial Trend Estimation (Degree 2) - Delhi Climate")
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```
### OUTPUT

<img width="571" height="399" alt="image" src="https://github.com/user-attachments/assets/ba3c9749-5579-4c38-b2f4-004f4242cfe1" />

<img width="494" height="302" alt="image" src="https://github.com/user-attachments/assets/54467cc4-6c74-47d8-94a7-bbe6da13559a" />


#### A - LINEAR TREND ESTIMATION
<img width="1102" height="452" alt="image" src="https://github.com/user-attachments/assets/e4bea95b-0663-4bc6-8896-a9124fbec974" />

#### B- POLYNOMIAL TREND ESTIMATION
<img width="1118" height="477" alt="image" src="https://github.com/user-attachments/assets/77b7eb2a-cd06-42a7-8cf3-130d7efb3096" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
