import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. DOWNLOAD DATA
# ============================================================

apple = yf.download(
    "AAPL",
    start="2015-01-01",
    end="2026-01-01"
)

market = yf.download(
    "^GSPC",
    start="2015-01-01",
    end="2026-01-01"
)


# ============================================================
# 2. CALCULATE DAILY RETURNS
# ============================================================

apple["Return"] = apple["Close"].pct_change()
market["Return"] = market["Close"].pct_change()

apple = apple.dropna()
market = market.dropna()


# ============================================================
# 3. APPLE SUMMARY STATISTICS
# ============================================================

mean_return = apple["Return"].mean()
volatility = apple["Return"].std()
minimum = apple["Return"].min()
maximum = apple["Return"].max()

print("\nApple Daily Return Statistics")
print("-----------------------------")
print(f"Mean daily return: {mean_return:.4%}")
print(f"Daily volatility:  {volatility:.4%}")
print(f"Minimum return:    {minimum:.4%}")
print(f"Maximum return:    {maximum:.4%}")


# ============================================================
# 4. CREATE REGRESSION DATASET
# ============================================================

data = pd.concat(
    [
        apple["Return"].rename("Apple_Return"),
        market["Return"].rename("Market_Return")
    ],
    axis=1
)

data = data.dropna()


# ============================================================
# 5. ADDITIONAL STATISTICS
# ============================================================

correlation = data["Apple_Return"].corr(data["Market_Return"])

annualised_return = mean_return * 252
annualised_volatility = volatility * (252 ** 0.5)

print("\nAdditional Statistics")
print("---------------------")
print(f"Apple/market correlation: {correlation:.4f}")
print(f"Annualised mean return:   {annualised_return:.2%}")
print(f"Annualised volatility:    {annualised_volatility:.2%}")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

train = data.loc[data.index < "2023-01-01"]
test = data.loc[data.index >= "2023-01-01"]

print("\nDataset Split")
print("-------------")
print(f"Training observations: {len(train)}")
print(f"Test observations:     {len(test)}")


# ============================================================
# 7. TRAIN THE REGRESSION MODEL
# ============================================================

X_train = train[["Market_Return"]]
y_train = train["Apple_Return"]

X_test = test[["Market_Return"]]
y_test = test["Apple_Return"]

model = LinearRegression()
model.fit(X_train, y_train)


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# ============================================================
# 9. REGRESSION RESULTS
# ============================================================

alpha = model.intercept_
beta = model.coef_[0]

train_r_squared = model.score(X_train, y_train)
test_r_squared = model.score(X_test, y_test)

print("\nRegression Results")
print("------------------")
print(f"Alpha:       {alpha:.6f}")
print(f"Beta:        {beta:.4f}")
print(f"Training R²: {train_r_squared:.4f}")
print(f"Test R²:     {test_r_squared:.4f}")


# ============================================================
# 10. TEST SET RESIDUALS
# ============================================================

test_residuals = y_test - test_predictions

residual_mean = test_residuals.mean()
residual_std = test_residuals.std()

print("\nTest Set Residual Statistics")
print("----------------------------")
print(f"Mean residual:                {residual_mean:.6f}")
print(f"Residual standard deviation: {residual_std:.6f}")


# ============================================================
# 11. PLOT APPLE'S RETURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    apple["Return"],
    bins=100
)

plt.title("Distribution of Apple Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig(
    "results/return_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 12. PLOT TEST SET REGRESSION
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["Market_Return"],
    y_test,
    alpha=0.3
)

sorted_indices = X_test["Market_Return"].argsort()

plt.plot(
    X_test["Market_Return"].iloc[sorted_indices],
    test_predictions[sorted_indices],
    linewidth=2,
    label="Regression line"
)

plt.title("Apple Returns vs S&P 500 Returns (Test Set)")
plt.xlabel("S&P 500 Daily Return")
plt.ylabel("Apple Daily Return")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 13. PLOT TEST SET RESIDUALS
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    test_predictions,
    test_residuals,
    alpha=0.3
)

plt.axhline(
    y=0,
    linewidth=2
)

plt.title("Test Set Regression Residuals")
plt.xlabel("Predicted Apple Return")
plt.ylabel("Residual")
plt.grid(True)

plt.savefig(
    "results/residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nPlots saved to the results folder.")
