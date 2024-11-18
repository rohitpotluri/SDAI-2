# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("cleaned_train_set.csv")

# Map sentiment labels to binary (1 for positive, 0 for negative)
data['label'] = data['label'].map({'__label__1': 0, '__label__2': 1})

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['review'], data['label'], test_size=0.2, random_state=42
)

# Text Tokenization and Padding
max_words = 20000  # Vocabulary size
max_len = 100      # Maximum review length (for padding)

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

# Convert texts to sequences and pad them
train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=max_len, padding='post', truncating='post')

# Load GloVe Embeddings
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip

embedding_dim = 100  # Using the 100-dimensional GloVe embeddings
embedding_index = {}

with open("glove.6B.100d.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Create Embedding Matrix
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build the LSTM Model
lstm_units = 64
dropout_rate = 0.5

model = Sequential([
    Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
    LSTM(lstm_units, return_sequences=False),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the Model
batch_size = 32
epochs = 5

history = model.fit(
    train_padded,
    train_labels,
    validation_data=(val_padded, val_labels),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)

model.save('lstm_sentiment_analysis_model.h5')
print("LSTM model saved as 'lstm_sentiment_analysis_model.h5'")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load test dataset
test_data = pd.read_csv("test_set.csv")

# Preprocess the test data
test_labels = test_data['label']
test_texts = test_data['text'].apply(lambda x: x.split(':')[1] if ':' in x else x)  # Assuming label format is '__label__X: Text'
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

# Evaluate the model on the new test data
predictions = model.predict(test_padded)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

# Print model summary
model.summary()

# Calculate metrics
accuracy = accuracy_score(test_labels, binary_predictions)
precision = precision_score(test_labels, binary_predictions)
recall = recall_score(test_labels, binary_predictions)
f1 = f1_score(test_labels, binary_predictions)
conf_matrix = confusion_matrix(test_labels, binary_predictions)
roc_auc = roc_auc_score(test_labels, predictions)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"ROC-AUC Score: {roc_auc}")

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Confusion Matrix')
plt.show()

# Plotting ROC curve
RocCurveDisplay.from_predictions(test_labels, predictions)
plt.title('ROC Curve')
plt.show()

