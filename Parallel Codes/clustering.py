import lightgbm as lgb
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, fbeta_score

import tensorflow as tf
import numpy as np
import pandas as pd


# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and TensorFlow is using it.")
else:
    print("No GPU found. Using CPU instead.")


# Enable memory growth for the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(e)


# Define the GPU and CPU devices
gpu_device = "/device:GPU:0"  # Adjust the index if you have multiple GPUs
cpu_device = "/device:CPU:0"

with tf.device(cpu_device):

    X = np.load('embeddings_1.npy')
    X_2 = np.load('embeddings_2.npy')
    X = np.concatenate((X, X_2), axis=0)

    num_rows = len(X)           # Number of rows
    num_columns = len(X[0])     # Number of columns (assuming non-empty and rectangular)
    print("X dimensions:", (num_rows, num_columns)) 

    # Step 1: Read label data from files (assuming you have already defined this part)
    label_data = []
    file_names = ['icd_codes_1.txt', 'icd_codes_2.txt']  # Update with actual filenames
    for file_name in file_names:
        with open(file_name, 'r') as file:
            label_data.extend(line.strip() for line in file if line.strip())

    # Step 2: Create a set of unique ICD-10 codes for efficient lookup
    unique_codes = set()
    for labels in label_data:
        unique_codes.update(labels.split(";"))
    unique_codes = sorted(unique_codes)  # Convert to a sorted list at the end

    # Step 3: Initialize the StringLookup layer
    lookup_layer = tf.keras.layers.StringLookup(vocabulary=unique_codes, output_mode="multi_hot", mask_token=None,num_oov_indices=0)

    # Step 4: Create a tf.data.Dataset to handle large data efficiently
    label_data_ds = tf.data.Dataset.from_tensor_slices(label_data)

        # Step 5: Define a function to encode each label set
    def encode_labels(labels):
        
        multi_hot = lookup_layer(tf.strings.split(labels, sep=";"))
        return tf.cast(multi_hot, dtype=tf.int16)  # Reducing the precision here

    # Step 6: Map encoding function over the dataset and batch it
    # Batch processing reduces memory usage
    multi_hot_labels_ds = label_data_ds.map(encode_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(1000)

    # Step 7: Concatenate all batches to get the final `y` tensor
    y = tf.concat(list(multi_hot_labels_ds), axis=0)

    # Ensure the correct shape of `y`
    print("Shape of y:", y.shape)  # Should output: (200000, 1400)

    y = y.numpy()

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a LightGBM classifier
    lgb_model = lgb.LGBMClassifier()

    # Wrap it in MultiOutputClassifier for multi-label support
    multi_target_model = MultiOutputClassifier(lgb_model, n_jobs=-1)
    multi_target_model.fit(X_train, y_train)

    # Make predictions
    y_pred = multi_target_model.predict(X_val)

    # Evaluate the model
    accuracy = fbeta_score(y_val, y_pred, beta=2, average='micro')
    print(f"Accuracy: {accuracy:.4f}")