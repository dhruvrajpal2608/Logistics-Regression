import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)
data = {
    'Distance_Miles': np.random.randint(100, 3000, 500),
    'Transit_Days_Allowed': np.random.randint(1, 10, 500),
}
df = pd.DataFrame(data)
# Create a dummy target variable with some random noise
df['Is_Late'] = (df['Distance_Miles'] / 300 - df['Transit_Days_Allowed'] + np.random.normal(0, 2, 500) > 0).astype(int)

# 2. Separate Features (X) and Target (y)
X = df[['Distance_Miles', 'Transit_Days_Allowed']]
y = df['Is_Late']

# 3. Split the data into Training and Testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling
# Logistic regression uses gradient descent or coordinate descent optimization;
# scaling ensures features contribute equally and helps convergence.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy Score: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
print(f"Intercept: {model.intercept_[0]:.4f}")



