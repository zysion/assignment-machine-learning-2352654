from pathlib import Path

import pandas as pd
import numpy as np
from data import read_data
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

# read data
X_train, X_val, y_train, y_val = read_data()

# Fit Decision Tree model
tree_model = DecisionTreeClassifier(random_state=31)
tree_model.fit(X_train, y_train)

# Extract parameters from the model
max_depth = tree_model.get_depth()
num_leaves = tree_model.get_n_leaves()
train_score = tree_model.score(X_train, y_train)
val_score = tree_model.score(X_val, y_val)

importances = tree_model.feature_importances_
feature_importance_list = list(enumerate(importances))
feature_importance_list.sort(key=lambda x: x[1], reverse=True)
top_3_features = [f"Column {feat[0]}" for feat in feature_importance_list[:3]]

print("=== RESULT tree ===")
print(f"1. Maximum depth: {max_depth}")
print(f"2. Number of leaves: {num_leaves}")
print(f"3. Training score (Accuracy): {train_score:.4f}")
print(f"4. Validation score (Accuracy): {val_score:.4f}")
print(f"5. Top 3 features (index): {top_3_features}")