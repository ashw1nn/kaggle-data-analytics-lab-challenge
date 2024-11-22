import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.initializers import he_uniform, he_normal, glorot_uniform, glorot_normal
from tensorflow.keras.models import clone_model
from tensorflow.keras.regularizers import l1, l2, l1_l2

@tf.keras.utils.register_keras_serializable()
class MicroF2Score(tf.keras.metrics.Metric):
    def __init__(self, name='micro_f2_score', beta=2, **kwargs):
        super(MicroF2Score, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Threshold y_pred to get binary predictions
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        # Cast y_true to float32 to ensure compatibility
        y_true = tf.cast(y_true, tf.float32)

        # Calculate true positives, false positives, and false negatives
        true_positive = tf.reduce_sum(y_true * y_pred)
        false_positive = tf.reduce_sum(y_pred * (1 - y_true))
        false_negative = tf.reduce_sum((1 - y_pred) * y_true)

        # Update the corresponding weights
        self.tp.assign_add(true_positive)
        self.fp.assign_add(false_positive)
        self.fn.assign_add(false_negative)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f_beta = (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall + tf.keras.backend.epsilon())
        return f_beta

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


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

with tf.device(cpu_device):
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

# Calculate inverse class frequencies
# epsilon = 1e-6  # To avoid division by zero
# label_counts = np.sum(y, axis=0) + epsilon  # Add epsilon to avoid zero counts
# total_labels = y.shape[0]

# # Calculate class weights based on inverse frequency
# class_weights = total_labels / (1400 * label_counts)

# # Calculate sample weights based on labels in each instance
# sample_weights = np.sum(y * class_weights, axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val, sample_weights_tr, sample_weights_val = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)



print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# # Define the model using the Sequential API with added complexity and batch normalization
# model = Sequential([
#     Input(shape=(X_train.shape[1],)),  # Specify the input shape directly
#     Dense(1024, activation='leaky_relu',),
#     BatchNormalization(),
#     # Dropout(0.2),
#     # Dense(512, activation='leaky_relu',),
#     # BatchNormalization(),
#     # Dropout(0.1),
#     Dense(1400, activation='sigmoid')  # Sigmoid activation for multi-label classification
# ])

with tf.device(gpu_device):
    # Compile model with Adam optimizer and custom focal loss
    optimizers = [Nadam]
    for optim in optimizers:
        # Define the model using the Sequential API with added complexity and batch normalization
        model = Sequential([
            Input(shape=(X_train.shape[1],)),  # Specify the input shape directly
            Dense(1024, activation='leaky_relu'),
            BatchNormalization(),
            # Dropout(0.2),
            # Dense(512, activation='leaky_relu',),
            # BatchNormalization(),
            # Dropout(0.1),
            Dense(1400, activation='sigmoid')  # Sigmoid activation for multi-label classification
        ])

        optimizer = optim(learning_rate=0.0003)  
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[MicroF2Score()])

        # Callbacks for early stopping and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, mode='min')
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, mode='min')

        # Train the model
        history = model.fit(X_train, y_train, 
                            batch_size=32,  # Optimized batch size
                            epochs=100,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, model_checkpoint, lr_scheduler],
                            verbose=1)
        model.save(f'model_{optim}.keras')