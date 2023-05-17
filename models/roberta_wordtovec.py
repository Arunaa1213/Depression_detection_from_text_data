import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf

# Load preprocessed data
preprocessed_data = pd.read_csv("preprocessed_data.csv")

# Load word2vec data
word2vec_data = pd.read_csv("word2vec_data.csv")

# Merge data on PID
merged_data = pd.merge(preprocessed_data, word2vec_data, on="PID")

# Split data into train and test sets
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Tokenize input data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
X_train = train_data['Text_data'].tolist()
X_train = tokenizer(X_train, padding=True, truncation=True, return_tensors='tf')
X_test = test_data['Text_data'].tolist()
X_test = tokenizer(X_test, padding=True, truncation=True, return_tensors='tf')

# Pad input data
max_length = max([len(text.split()) for text in preprocessed_data['Text_data'].tolist()])
X_train = pad_sequences(X_train['input_ids'], maxlen=max_length, dtype="long", value=0, truncating="post", padding="post")
X_test = pad_sequences(X_test['input_ids'], maxlen=max_length, dtype="long", value=0, truncating="post", padding="post")

# Convert labels to one-hot encoded vectors
y_train = preprocessed_data['Label']
y_train = [0 if label == 'not depression' else 1 if label == 'moderate' else 2 for label in y_train]
y_train = to_categorical(y_train, num_classes=3)
y_test = preprocessed_data['Label']
y_test = [0 if label == 'not depression' else 1 if label == 'moderate' else 2 for label in y_test]
y_test = to_categorical(y_test, num_classes=3)

# Load RoBERTa model
roberta = TFRobertaModel.from_pretrained('roberta-base')

# Create model architecture
input_layer = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
embedding_layer = roberta.roberta(input_layer)[0]
pooling_layer = tf.keras.layers.GlobalMaxPool1D()(embedding_layer)
dense_layer = tf.keras.layers.Dense(128, activation='relu')(pooling_layer)
output_layer = tf.keras.layers.Dense(3, activation='softmax')(dense_layer)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
