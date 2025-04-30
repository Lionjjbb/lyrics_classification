# evaluate_performance.py

import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to your files (update if necessary)
test_file = 'data/libsvm_test.txt'
predictions_file = 'predictions.txt'
mapping_file = 'tag_to_label_mapping.txt'

# Step 1: Read the true labels from the test file
true_labels = []
with open(test_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            label = int(line.split()[0])
            true_labels.append(label)

# Step 2: Read the predicted labels from the predictions file
predicted_labels = []
with open(predictions_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            label = int(float(line))
            predicted_labels.append(label)

# Step 3: Load the label to category mapping
label_to_tag = {}
with open(mapping_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                label_str, tag = line.split(': ', 1)
                label = int(label_str)
                label_to_tag[label] = tag
            except ValueError:
                continue  # skip lines that don't match the expected format

# Ensure the labels are aligned
assert len(true_labels) == len(predicted_labels), "Mismatch in number of labels"

# Get the list of labels and their corresponding category names
labels = sorted(label_to_tag.keys())
target_names = [label_to_tag[label] for label in labels]

# Step 4: Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Step 5: Generate a classification report
report = classification_report(true_labels, predicted_labels, labels=labels, target_names=target_names)
print("\nClassification Report:")
print(report)

# Optional: Visualize the confusion matrix
try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Seaborn or Matplotlib is not installed. Skipping confusion matrix visualization.")