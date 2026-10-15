"""
Module 2: Polynomial and Quadratic Regression for Solar Power Forecasting
Fits Degree 2 (Quadratic) and Degree 3 Polynomial Models and compares with Linear Regression.
Generates publication-quality charts saved to outputs/Polynomial_Regression_M2.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "Plant_1_Merged_Data.csv"
OUT_DIR = BASE_DIR / "outputs" / "Polynomial_Regression_M2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dark theme palette matching Solar Sense
BG_COLOR = "#0c121e"
CARD_BG = "#131c2e"
TEXT_COLOR = "#e2e8f0"
ACCENT_CYAN = "#06b6d4"
ACCENT_AMBER = "#f59e0b"
ACCENT_EMERALD = "#10b981"
ACCENT_PURPLE = "#a855f7"
GRID_COLOR = "#1e293b"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "font.sans-serif": "DejaVu Sans",
})

print("Loading dataset...")
df = pd.read_csv(DATA_PATH).dropna(subset=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DC_POWER"])

# Filter daylight records
daylight_df = df[df["IRRADIATION"] > 0.005].copy()
features = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
X = daylight_df[features].values
y = daylight_df["DC_POWER"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Model
linear_model = LinearRegression().fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_lin = r2_score(y_test, y_pred_linear)
rmse_lin = float(np.sqrt(mean_squared_error(y_test, y_pred_linear)))
mae_lin = float(mean_absolute_error(y_test, y_pred_linear))

# 2. Quadratic (Degree 2)
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)
poly2_model = LinearRegression().fit(X_train_poly2, y_train)
y_pred_poly2 = poly2_model.predict(X_test_poly2)
r2_poly2 = r2_score(y_test, y_pred_poly2)
rmse_poly2 = float(np.sqrt(mean_squared_error(y_test, y_pred_poly2)))
mae_poly2 = float(mean_absolute_error(y_test, y_pred_poly2))

# 3. Cubic (Degree 3)
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)
poly3_model = LinearRegression().fit(X_train_poly3, y_train)
y_pred_poly3 = poly3_model.predict(X_test_poly3)
r2_poly3 = r2_score(y_test, y_pred_poly3)
rmse_poly3 = float(np.sqrt(mean_squared_error(y_test, y_pred_poly3)))
mae_poly3 = float(mean_absolute_error(y_test, y_pred_poly3))

# Metrics DataFrame
metrics_df = pd.DataFrame([
    {"Model": "Linear Regression (Deg 1)", "Degree": 1, "R2_Score": round(r2_lin, 4), "RMSE": round(rmse_lin, 2), "MAE": round(mae_lin, 2), "Terms": X_train.shape[1]},
    {"Model": "Quadratic Regression (Deg 2)", "Degree": 2, "R2_Score": round(r2_poly2, 4), "RMSE": round(rmse_poly2, 2), "MAE": round(mae_poly2, 2), "Terms": X_train_poly2.shape[1]},
    {"Model": "Cubic Polynomial (Deg 3)", "Degree": 3, "R2_Score": round(r2_poly3, 4), "RMSE": round(rmse_poly3, 2), "MAE": round(mae_poly3, 2), "Terms": X_train_poly3.shape[1]},
])
metrics_df.to_csv(OUT_DIR / "poly_metrics.csv", index=False)
print("Metrics saved:\n", metrics_df)

# Feature names and quadratic formula
feature_names_poly2 = poly2.get_feature_names_out(features)
poly_terms = [f"{coef:+.4f} × {name}" for coef, name in zip(poly2_model.coef_, feature_names_poly2)]
equation_str = f"DC_POWER = {poly2_model.intercept_:.4f}\n  " + "\n  ".join(poly_terms)
with open(OUT_DIR / "poly_equation.txt", "w", encoding="utf-8") as f:
    f.write(equation_str)

# Save coefficients
coef_df = pd.DataFrame({
    "Term": feature_names_poly2,
    "Coefficient": poly2_model.coef_,
    "Abs_Weight": np.abs(poly2_model.coef_)
}).sort_values(by="Abs_Weight", ascending=False)
coef_df.to_csv(OUT_DIR / "poly2_coefficients.csv", index=False)

# -------------------------------------------------------------
# Chart 1: Curvature & Response Curve (Irradiation vs DC Power)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
sample_idx = np.random.choice(len(daylight_df), size=min(1500, len(daylight_df)), replace=False)
sample_df = daylight_df.iloc[sample_idx]

ax.scatter(sample_df["IRRADIATION"], sample_df["DC_POWER"], color="#64748b", alpha=0.35, s=18, label="Actual Telemetry Observations")

# Generate smooth sweep for irradiation while keeping ambient & module temp at medians
irr_sweep = np.linspace(0.01, 1.25, 200)
amb_med = daylight_df["AMBIENT_TEMPERATURE"].median()
mod_med = daylight_df["MODULE_TEMPERATURE"].median()
sweep_matrix = np.column_stack([
    np.full_like(irr_sweep, amb_med),
    np.full_like(irr_sweep, mod_med),
    irr_sweep
])

# Predictions on sweep
sweep_lin = linear_model.predict(sweep_matrix)
sweep_poly2 = poly2_model.predict(poly2.transform(sweep_matrix))
sweep_poly3 = poly3_model.predict(poly3.transform(sweep_matrix))

ax.plot(irr_sweep, sweep_lin, color=ACCENT_AMBER, linestyle="--", linewidth=2.5, label=f"Linear Fit (R² = {r2_lin:.3f})")
ax.plot(irr_sweep, sweep_poly2, color=ACCENT_CYAN, linewidth=3.2, label=f"Quadratic Deg 2 Fit (R² = {r2_poly2:.3f})")
ax.plot(irr_sweep, sweep_poly3, color=ACCENT_EMERALD, linestyle=":", linewidth=2.5, label=f"Cubic Deg 3 Fit (R² = {r2_poly3:.3f})")

ax.set_title("Solar Power Curve: Linear vs Quadratic vs Cubic Polynomial Fitting", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Solar Irradiation (kW/m²)", fontsize=11, fontweight="600")
ax.set_ylabel("Generated DC Power (kW)", fontsize=11, fontweight="600")
ax.grid(True)
ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper left", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "polynomial_quadratic_curve.png", dpi=150)
plt.close(fig)

# -------------------------------------------------------------
# Chart 2: Actual vs Predicted Scatter (Degree 2 Quadratic)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
plot_idx = np.random.choice(len(y_test), size=min(1200, len(y_test)), replace=False)
y_t_sub = y_test[plot_idx]
y_p_sub = y_pred_poly2[plot_idx]

scatter = ax.scatter(y_t_sub, y_p_sub, c=np.abs(y_t_sub - y_p_sub), cmap="plasma", alpha=0.6, s=22)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Absolute Error |y - ŷ| (kW)", color=TEXT_COLOR)
cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)

# Diagonal reference line
min_val = min(y_t_sub.min(), y_p_sub.min())
max_val = max(y_t_sub.max(), y_p_sub.max())
ax.plot([min_val, max_val], [min_val, max_val], color=ACCENT_EMERALD, linestyle="--", linewidth=2, label="Perfect Forecast (1:1 Ideal)")

ax.set_title(f"Quadratic Regression: Actual vs. Predicted (R² = {r2_poly2:.4f}, RMSE = {rmse_poly2:.1f})", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Actual DC Power (kW)", fontsize=11, fontweight="600")
ax.set_ylabel("Quadratic Predicted DC Power (kW)", fontsize=11, fontweight="600")
ax.grid(True)
ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper left", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "actual_vs_predicted_poly.png", dpi=150)
plt.close(fig)

# -------------------------------------------------------------
# Chart 3: Residuals Comparison (Linear vs Quadratic)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

res_lin = y_test - y_pred_linear
res_poly2 = y_test - y_pred_poly2

ax1.scatter(y_pred_linear[plot_idx], res_lin[plot_idx], color=ACCENT_AMBER, alpha=0.4, s=16)
ax1.axhline(0, color="#ef4444", linestyle="--", linewidth=1.5)
ax1.set_title(f"Linear Residuals (RMSE: {rmse_lin:.1f})", fontweight="bold", fontsize=12)
ax1.set_xlabel("Predicted DC Power (kW)")
ax1.set_ylabel("Residual Error (Actual - Pred)")
ax1.grid(True)

ax2.scatter(y_pred_poly2[plot_idx], res_poly2[plot_idx], color=ACCENT_CYAN, alpha=0.4, s=16)
ax2.axhline(0, color="#ef4444", linestyle="--", linewidth=1.5)
ax2.set_title(f"Quadratic Deg 2 Residuals (RMSE: {rmse_poly2:.1f})", fontweight="bold", fontsize=12)
ax2.set_xlabel("Predicted DC Power (kW)")
ax2.set_ylabel("Residual Error (Actual - Pred)")
ax2.grid(True)

plt.suptitle("Residual Diagnostics: Elimination of Non-Linear Curvature Error", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "residuals_poly.png", dpi=150)
plt.close(fig)

print("All Polynomial & Quadratic artifacts successfully generated in:", OUT_DIR)
