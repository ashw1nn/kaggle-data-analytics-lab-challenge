import tensorflow as tf
import numpy as np
import pandas as pd

    

@tf.keras.utils.register_keras_serializable()  # Register the class
class MicroF2Score(tf.keras.metrics.Metric):
    def _init_(self, name='micro_f2_score', beta=2, **kwargs):
        super(MicroF2Score, self)._init_(name=name, **kwargs)
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
        f_beta = (1 + self.beta*2) * (precision * recall) / (self.beta*2 * precision + recall + tf.keras.backend.epsilon())
        return f_beta

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


with tf.device('/device:CPU:0'):

    model = tf.keras.models.load_model('model.keras')

    X = np.load('embeddings_1.npy')
    X_2 = np.load('embeddings_2.npy')
    X = np.concatenate((X, X_2), axis=0)
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


with tf.device('/device:CPU:0'):
    # Load best model
    # model = tf.keras.models.load_model('best_model.keras')
    # Load test data
    X_test = np.load('test_data.npy')

    # Predict probabilities and convert to binary using the tuned thresholds
    # y_test_pred = predict_with_thresholds(model, X_test, best_thresholds)
    y_test_pred = model.predict(X_test)

    # Prepare reverse lookup
    lookup_layer = tf.keras.layers.StringLookup(
        vocabulary=unique_codes, invert=True, output_mode="int", mask_token=None, num_oov_indices=0
    )

    # Convert predictions to ICD10 codes
    predicted_indices = [np.where(pred_row == 1)[0] for pred_row in y_test_pred]
    predicted_codes = [lookup_layer(indices).numpy() for indices in predicted_indices]
    predicted_codes = [[code.decode('utf-8') for code in row] for row in predicted_codes]
    predicted_labels = [';'.join(row) for row in predicted_codes]

    # Create and save submission DataFrame
    submission_df = pd.DataFrame({
        'id': range(1, len(predicted_labels) + 1),
        'labels': predicted_labels
    })
    submission_df.to_csv('submission_srini.csv', index=False)
