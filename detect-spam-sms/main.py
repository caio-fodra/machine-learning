import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow_hub as hub

df = pd.read_csv("spam.csv", encoding='latin-1')
df.head()

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'Text'})
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
df.head()

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'],
    df['label_enc'],
    test_size=0.2,
    random_state=42
)

# --- CRITICAL FIX: Convert to NumPy Arrays ---
# Keras 3+ throws "Invalid dtype: object" if you pass Pandas Series directly.
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Calculate stats for vectorization
avg_words_len = round(sum([len(i.split()) for i in df['Text']]) / len(df['Text']))
total_words_length = len(set(" ".join(df['Text']).split()))

print(f"Data Loaded. Training samples: {len(X_train_np)}")

# Helper function to compile and fit
def compile_and_fit(model, epochs=5):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Using the NumPy arrays (_np) here to prevent errors
    history = model.fit(X_train_np, y_train_np, epochs=epochs,
                        validation_data=(X_test_np, y_test_np))
    return history

# Helper function for metrics
def get_metrics(model, X, y):
    y_preds = np.round(model.predict(X))
    return {
        'accuracy': accuracy_score(y, y_preds),
        'precision': precision_score(y, y_preds),
        'recall': recall_score(y, y_preds),
        'f1-score': f1_score(y, y_preds)
    }

# --- Text Vectorization Layer ---
from tensorflow.keras.layers import TextVectorization

text_vec = TextVectorization(
    max_tokens=total_words_length,
    standardize='lower_and_strip_punctuation',
    output_mode='int',
    output_sequence_length=avg_words_len
)

# Adapt to the training data (NumPy array)
text_vec.adapt(X_train_np)

avg_words_len = round(sum([len(i.split())
                      for i in df['Text']]) / len(df['Text']))
total_words_length = len(set(" ".join(df['Text']).split()))

print(f"Data Loaded. Training samples: {len(X_train_np)}")
print(f"Average words per message: {avg_words_len}")
print(f"Approximate vocabulary size: {total_words_length}")

# Inputs
input_layer = layers.Input(shape=(1,), dtype=tf.string)

# Vectorization
x = text_vec(input_layer)

# Embedding (Removed input_length argument)
x = layers.Embedding(input_dim=total_words_length, output_dim=128)(x)

# Architecture
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(1, activation='sigmoid')(x)

# Compile & Train
model_1 = keras.Model(input_layer, output_layer, name="Dense_Model")
history_1 = compile_and_fit(model_1)

# Inputs
input_layer = layers.Input(shape=(1,), dtype=tf.string)

# Vectorization
x = text_vec(input_layer)

# Embedding (Removed input_length argument)
x = layers.Embedding(input_dim=total_words_length, output_dim=128)(x)

# LSTM Layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)

# Architecture
x = layers.Flatten()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(1, activation='sigmoid')(x)

# Compile & Train
model_2 = keras.Model(input_layer, output_layer, name="BiLSTM_Model")
history_2 = compile_and_fit(model_2)

# 1. Load the Layer
use_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    trainable=False,
    input_shape=[],
    dtype=tf.string,
    name='USE'
)

# 2. Build Model using Functional API
# We use shape=[] because USE expects a list of strings (scalars), not a vector of strings
input_layer = layers.Input(shape=[], dtype=tf.string)

# CRITICAL FIX: Add 'output_shape=(512,)'
# We tell Keras: "Trust us, this function will return a vector of size 512"
embedding = layers.Lambda(lambda x: use_layer(x), output_shape=(512,))(input_layer)

x = layers.Dense(64, activation='relu')(embedding)
x = layers.Dropout(0.2)(x)
output_layer = layers.Dense(1, activation='sigmoid')(x)

model_3 = keras.Model(input_layer, output_layer, name="USE_Model")

# 3. Compile and Train
history_3 = compile_and_fit(model_3)

# ---------------------------------------------------------
# 1. Gather Results
# ---------------------------------------------------------
results = {
    'Dense Embedding': get_metrics(model_1, X_test_np, y_test_np),
    'Bi-LSTM': get_metrics(model_2, X_test_np, y_test_np),
    'Transfer Learning (USE)': get_metrics(model_3, X_test_np, y_test_np)
}

results_df = pd.DataFrame(results).transpose()
print("Performance Table:")
print(results_df)

# ---------------------------------------------------------
# 2. Bar Chart Comparison
# ---------------------------------------------------------
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Metrics (Bar Chart)")
plt.ylabel("Score")
plt.ylim(0.8, 1.0) # Zoom in to see differences
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ---------------------------------------------------------
# 3. Line Graph Comparison
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot a line for each model
# We transpose the DF so X-axis = Metrics, Lines = Models
for model_name in results_df.index:
    plt.plot(results_df.columns, results_df.loc[model_name],
             marker='o', label=model_name, linewidth=2)

plt.title("Model Performance Trends (Line Graph)")
plt.ylabel("Score")
plt.xlabel("Metric")
plt.ylim(0.8, 1.0) # Zoom in on the top range
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
