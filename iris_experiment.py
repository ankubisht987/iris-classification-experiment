import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Step 1: Load Dataset ---
# We are using the Iris dataset (a classic classification problem)
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for a cleaner look at the data (optional, but good for analysis)
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Dataset Preview:")
print(df.head())
print("-" * 30)

# --- Step 2: Data Preprocessing ---
# Split the data: 80% for training the model, 20% for testing it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape:  {X_test.shape}")
print("-" * 30)

# --- Step 3: Build & Train Model ---
# We use a Decision Tree because it is easy to interpret and visualize
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

print("Model training complete.")
print("-" * 30)

# --- Step 4: Evaluation ---
# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --- Step 5: Visualization ---
# 1. Confusion Matrix (To see where errors happened)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 2. Decision Tree Plot (To see the logic)
plt.subplot(1, 2, 2)
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree Logic')

plt.tight_layout()
plt.show()