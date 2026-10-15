import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix




# ============================================================
# 1. SETTINGS
# ============================================================


project_root = Path(__file__).resolve().parent.parent
DATASET = project_root / "dataset" / "final_preprocess_M2.csv"
OUTPUT_FOLDER = project_root / "outputs" / "Logistic_Regression_Binary_Classify_M2"


os.makedirs(OUTPUT_FOLDER, exist_ok=True)




# ============================================================
# 2. LOAD SOLAR DATASET
# ============================================================


df = pd.read_csv(DATASET)


print("\n========== DATASET ==========")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())


print("\nDataset shape:")
print(df.shape)




# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

# Create binary classification target: Is_Daytime
# (1 if IRRADIATION > min_irradiation, else 0)
min_irrad = df["IRRADIATION"].min()
df["Is_Daytime"] = (df["IRRADIATION"] > min_irrad).astype(int)

FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "DAILY_YIELD"]
TARGET = "Is_Daytime"


X = df[FEATURES].values
y = df[TARGET].values




# Make sure target is integer 0/1
y = y.astype(int)




print("\nFeatures:", FEATURES)
print("Target:", TARGET)


print("\nTarget distribution:")
print(pd.Series(y).value_counts())




# ============================================================
# 4. SPLIT DATA
# ============================================================


X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.25,
   random_state=42,
   stratify=y
)




# ============================================================
# 5. STANDARDIZE FEATURES
# ============================================================


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# ============================================================
# 6. SIGMOID FUNCTION
# ============================================================


def sigmoid(z):
   """
   Sigmoid:


            1
   σ(z) = -------
          1 + e^(-z)


   Converts the linear score into probability.
   """


   z = np.clip(z, -500, 500)


   return 1 / (1 + np.exp(-z))




# ============================================================
# 7. SIGMOID GRAPH
# ============================================================


z_values = np.linspace(-10, 10, 500)
sigmoid_values = sigmoid(z_values)


plt.figure(figsize=(8, 6))


plt.plot(
   z_values,
   sigmoid_values,
   color="blue",
   linewidth=3
)


plt.axhline(
   0.5,
   color="red",
   linestyle="--",
   label="Threshold = 0.5"
)


plt.axvline(
   0,
   color="black",
   linestyle="--"
)


plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.legend()
plt.grid(alpha=0.3)


plt.savefig(
   os.path.join(
       OUTPUT_FOLDER,
       "01_sigmoid.png"
   ),
   dpi=300,
   bbox_inches="tight"
)

plt.close()




# ============================================================
# 8. CROSS-ENTROPY LOSS
# ============================================================


def cross_entropy_loss(y_true, y_probability):
   """
   Binary Cross-Entropy Loss:


   L = -1/m Σ [
           y log(p) + (1-y) log(1-p)
       ]
   """


   epsilon = 1e-15


   y_probability = np.clip(
       y_probability,
       epsilon,
       1 - epsilon
   )


   loss = -np.mean(
       y_true * np.log(y_probability)
       +
       (1 - y_true) * np.log(1 - y_probability)
   )


   return loss




# ============================================================
# 9. INITIALIZE LOGISTIC REGRESSION
# ============================================================


number_of_features = X_train_scaled.shape[1]


weights = np.zeros(number_of_features)


bias = 0.0


learning_rate = 0.05


epochs = 3000


loss_history = []




# ============================================================
# 10. TRAIN USING GRADIENT DESCENT
# ============================================================


m = len(y_train)


for epoch in range(epochs):


   # --------------------------------------------------------
   # Linear model
   #
   # z = w1*x1 + w2*x2 + w3*x3 + b
   # --------------------------------------------------------


   z = np.dot(
       X_train_scaled,
       weights
   ) + bias


   # --------------------------------------------------------
   # Sigmoid
   # --------------------------------------------------------


   probability = sigmoid(z)


   # --------------------------------------------------------
   # Cross-entropy loss
   # --------------------------------------------------------


   loss = cross_entropy_loss(
       y_train,
       probability
   )


   loss_history.append(loss)


   # --------------------------------------------------------
   # Gradients
   # --------------------------------------------------------


   error = probability - y_train


   dw = (
       1 / m
   ) * np.dot(
       X_train_scaled.T,
       error
   )


   db = (
       1 / m
   ) * np.sum(error)


   # --------------------------------------------------------
   # Update parameters
   # --------------------------------------------------------


   weights -= learning_rate * dw


   bias -= learning_rate * db




# ============================================================
# 11. DISPLAY MODEL PARAMETERS
# ============================================================


print("\n========== MODEL PARAMETERS ==========")


for feature, weight in zip(FEATURES, weights):
   print(f"{feature}: {weight:.6f}")


print(f"Bias: {bias:.6f}")




# ============================================================
# 12. CROSS-ENTROPY LOSS GRAPH
# ============================================================


plt.figure(figsize=(8, 6))


plt.plot(
   range(1, epochs + 1),
   loss_history,
   color="purple",
   linewidth=2
)


plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")


plt.title(
   "Logistic Regression - Cross-Entropy Loss"
)


plt.grid(alpha=0.3)


plt.savefig(
   os.path.join(
       OUTPUT_FOLDER,
       "02_cross_entropy_loss.png"
   ),
   dpi=300,
   bbox_inches="tight"
)

plt.close()




# ============================================================
# 13. PREDICTION FUNCTIONS
# ============================================================


def predict_probability(X):
   """
   Calculate placement probability.
   """


   z = np.dot(X, weights) + bias


   return sigmoid(z)




def predict(X, threshold=0.5):
   """
   Convert probability into binary class.


   probability >= 0.5 -> DC_POWER
   probability < 0.5  -> Not DC_POWER
   """


   probability = predict_probability(X)


   return (probability >= threshold).astype(int)




# ============================================================
# 14. TEST SET PREDICTION
# ============================================================


test_probability = predict_probability(
   X_test_scaled
)


y_pred = predict(
   X_test_scaled
)




# ============================================================
# 15. ACCURACY
# ============================================================


accuracy = accuracy_score(
   y_test,
   y_pred
)


print("\n========== MODEL PERFORMANCE ==========")


print(
   f"Accuracy: {accuracy * 100:.2f}%"
)




# ============================================================
# 16. CONFUSION MATRIX
# ============================================================


cm = confusion_matrix(
   y_test,
   y_pred
)


print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(7, 6))


plt.imshow(
   cm,
   cmap="Blues"
)


plt.colorbar()


plt.xticks(
   [0, 1],
   ["Nighttime", "Daytime"]
)


plt.yticks(
   [0, 1],
   ["Nighttime", "Daytime"]
)


plt.xlabel("Predicted")
plt.ylabel("Actual")


plt.title("Confusion Matrix")


for i in range(2):
   for j in range(2):


       plt.text(
           j,
           i,
           cm[i, j],
           ha="center",
           va="center",
           fontsize=16
       )


plt.savefig(
   os.path.join(
       OUTPUT_FOLDER,
       "03_confusion_matrix.png"
   ),
   dpi=300,
   bbox_inches="tight"
)

plt.close()




# ============================================================
# 17. DECISION BOUNDARY AS HYPERPLANE
# ============================================================
#
# Logistic regression equation:
#
# z = w1*x1 + w2*x2 + w3*x3 + b
#
# The decision boundary occurs at:
#
# probability = 0.5
#
# sigmoid(z) = 0.5
#
# Therefore:
#
# z = 0
#
# Hence:
#
# w1*x1 + w2*x2 + w3*x3 + b = 0
#
# This is a HYPERPLANE.
#
# Since we have 3 features, the complete boundary
# exists in 3-dimensional feature space.
#
# We visualize a 2D cross-section by fixing
# Internship = 0.




# ============================================================
# 18. CREATE GRID FOR DECISION BOUNDARY
# ============================================================


temp_values = np.linspace(
   df["AMBIENT_TEMPERATURE"].min() - 0.2,
   df["AMBIENT_TEMPERATURE"].max() + 0.2,
   300
)


mod_temp_values = np.linspace(
   df["MODULE_TEMPERATURE"].min() - 0.2,
   df["MODULE_TEMPERATURE"].max() + 0.2,
   300
)


AMBIENT_TEMPERATURE, MODULE_TEMPERATURE = np.meshgrid(
   temp_values,
   mod_temp_values
)




# Fix DAILY_YIELD = 0 (its standardized mean)
DAILY_YIELD_FIXED = np.zeros_like(AMBIENT_TEMPERATURE)




# Create grid matching the 3 features
grid = np.column_stack([
   AMBIENT_TEMPERATURE.ravel(),
   MODULE_TEMPERATURE.ravel(),
   DAILY_YIELD_FIXED.ravel()
])




# Standardize grid
grid_scaled = scaler.transform(grid)




# Calculate probability
grid_probability = predict_probability(
   grid_scaled
)


grid_probability = grid_probability.reshape(
   AMBIENT_TEMPERATURE.shape
)




# ============================================================
# 19. PLOT DECISION BOUNDARY
# ============================================================


plt.figure(figsize=(10, 7))


# Probability regions
contour = plt.contourf(
   AMBIENT_TEMPERATURE,
   MODULE_TEMPERATURE,
   grid_probability,
   levels=50,
   cmap="RdYlGn",
   alpha=0.35
)


plt.colorbar(
   contour,
   label="Is_Daytime Probability"
)




# Hyperplane cross-section
plt.contour(
   AMBIENT_TEMPERATURE,
   MODULE_TEMPERATURE,
   grid_probability,
   levels=[0.5],
   colors="black",
   linewidths=3
)




# Original data
plt.scatter(
   df[df[TARGET] == 0]["AMBIENT_TEMPERATURE"],
   df[df[TARGET] == 0]["MODULE_TEMPERATURE"],
   color="red",
   edgecolor="black",
   s=60,
   label="Nighttime (Offline)"
)


plt.scatter(
   df[df[TARGET] == 1]["AMBIENT_TEMPERATURE"],
   df[df[TARGET] == 1]["MODULE_TEMPERATURE"],
   color="green",
   edgecolor="black",
   s=60,
   label="Daytime (Online)"
)


plt.xlabel("AMBIENT_TEMPERATURE")
plt.ylabel("MODULE_TEMPERATURE")


plt.title(
   "Logistic Regression Decision Boundary\n"
   "w₁(AMBIENT_TEMP) + w₂(MODULE_TEMP) + w₃(DAILY_YIELD) + b = 0"
)


plt.legend()
plt.grid(alpha=0.2)


plt.savefig(
   os.path.join(
       OUTPUT_FOLDER,
       "04_decision_boundary_hyperplane.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 20. PREDICT NEW SOLAR STATE
# ============================================================


# Example solar state:
# AMBIENT_TEMPERATURE = 0.5 (scaled)
# MODULE_TEMPERATURE = 0.8 (scaled)
# DAILY_YIELD = 0.0 (scaled)


new_state = np.array([
   [0.5, 0.8, 0.0]
])


new_state_scaled = scaler.transform(
   new_state
)


new_probability = predict_probability(
   new_state_scaled
)[0]


new_prediction = int(
   new_probability >= 0.5
)




print("\n========== NEW STATE PREDICTION ==========")


print("AMBIENT_TEMPERATURE:", new_state[0][0])
print("MODULE_TEMPERATURE:", new_state[0][1])
print("DAILY_YIELD:", new_state[0][2])


print(
   "Is_Daytime Probability:",
   f"{new_probability * 100:.2f}%"
)


if new_prediction == 1:
   print("Prediction: DAYTIME (ONLINE)")
else:
   print("Prediction: NIGHTTIME (OFFLINE)")




# ============================================================
# 21. SAVE PREDICTION RESULTS
# ============================================================


results = pd.DataFrame({
   "Actual": y_test,
   "Predicted": y_pred,
   "Is_Daytime_Probability": test_probability
})


results.to_csv(
   os.path.join(
       OUTPUT_FOLDER,
       "prediction_results.csv"
   ),
   index=False
)




# ============================================================
# 22. FINISH
# ============================================================


print("\n============================================")
print("PROGRAM COMPLETED")
print("============================================")


print(
   "\nAll output images are stored in:"
)


print(
   os.path.abspath(OUTPUT_FOLDER)
)


print("\nGenerated images:")


print("1. 01_sigmoid.png")
print("2. 02_cross_entropy_loss.png")
print("3. 03_confusion_matrix.png")
print("4. 04_decision_boundary_hyperplane.png")


print("\nPrediction results:")
print("5. prediction_results.csv")
