# svm_classification_top3.py

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load processed data
print("Loading processed data...")
X = sparse.load_npz('data/X_top3.npz')
y = pd.read_csv('data/y_top3.csv')['label'].values

# Load mapping from labels to tags
print("Loading label mapping...")
label_to_tag = {}
with open('tag_to_label_mapping.txt', 'r') as f:
    for line in f:
        label_str, tag = line.strip().split(': ')
        label_to_tag[int(label_str)] = tag

target_names = [label_to_tag[i] for i in sorted(label_to_tag.keys())]

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [1],  # Reduced range
    'kernel': ['linear', 'rbf'],  # Try both kernel
    'class_weight': [None]
}

# Set up stratified k-fold cross-validation with fewer folds
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model
print("Starting grid search...")
grid_search.fit(X, y)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate the best model on the whole dataset using cross-validation
from sklearn.model_selection import cross_val_predict

best_model = grid_search.best_estimator_
y_pred = cross_val_predict(best_model, X, y, cv=kf)
accuracy = accuracy_score(y, y_pred)
print(f"\nOverall accuracy with best SVM model: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=target_names))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y, y_pred)
print(cm)

# Optional: Visualize the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Top 3 Categories')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the best model
print("Saving the best SVM model...")
with open('best_svm_model_top3.pkl', 'wb') as f:
    pickle.dump(best_model, f)