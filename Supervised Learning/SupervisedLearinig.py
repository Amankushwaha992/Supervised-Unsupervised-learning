import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load CSV
data = pd.read_csv("knn_dataset.csv")

# Features & target
X = data[['age', 'bmi']].values # Corrected 'salary' to 'bmi'
y = data['label'].values # Corrected 'bought' to 'label'

# Scale features (IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Create mesh grid
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict grid points
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot actual data points
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k')

# New test point (adjusted for 'age' and 'bmi')
new_point = np.array([[30, 25]]) # Example: age 30, bmi 25
new_point_scaled = scaler.transform(new_point)

prediction = knn.predict(new_point_scaled)

# Plot test point
plt.scatter(new_point_scaled[:, 0], new_point_scaled[:, 1],
            marker='x', s=200)

plt.title(f"KNN Visualization (Predicted Class: {prediction[0]})")
plt.xlabel("Age (scaled)")
plt.ylabel("BMI (scaled)") # Updated label
plt.show()