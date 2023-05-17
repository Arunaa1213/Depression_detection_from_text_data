import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load the preprocessed FastText embedding data
fasttext_data = pd.read_csv('../fasttext_data.csv')

# Define the input and output columns
X = fasttext_data['PID'].values
y = fasttext_data['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RoBERTa tokenizer and model
tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
roberta_model = tf.keras.models.load_model('roberta_model')

# Tokenize the text data and convert the labels to categorical
X_train_tokenized = tokenizer.sequences_to_texts(X_train)
X_test_tokenized = tokenizer.sequences_to_texts(X_test)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Pad the tokenized text data to a maximum length
max_len = 100
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_tokenized, maxlen=max_len, padding='post')
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_tokenized, maxlen=max_len, padding='post')

# Define the model architecture
input_layer = Input(shape=(max_len,))
roberta_layer = roberta_model(input_layer)[0]
dense_layer = Dense(64, activation='relu')(roberta_layer)
dropout_layer = Dropout(0.5)(dense_layer)
output_layer = Dense(2, activation='softmax')(dropout_layer)
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with appropriate optimizer and loss function
optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(X_train_padded, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test_padded, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
