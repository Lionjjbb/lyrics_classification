#!/bin/bash

# Export variables for data paths
export TRAIN_DOCS="data/train.csv"
export TEST_DOCS="data/test.csv"

# Paths for LibSVM formatted data
export LIBSVM_TRAIN_DOCS="data/libsvm_train.txt"
export LIBSVM_TEST_DOCS="data/libsvm_test.txt"

# Optional: Model and predictions paths
export LIBSVM_MODEL="results/train.model"
export PREDICTION_RESULTS="results/predictions.txt"
