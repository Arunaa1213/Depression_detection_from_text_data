import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel

# Load the preprocessed data
preprocessed_data = pd.read_csv('preprocessed_data.csv')

# Load the FastText embedded data
fasttext_data = pd.read_csv('fasttext_data.csv')

# Convert label column to categorical
labels = fasttext_data['Label']
y = to_categorical(labels)

# Tokenize the text data using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

X = preprocessed_data['Text_data'].values
X = ["[CLS] " + str(sentence) + " [SEP]" for sentence in X]
X_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=512) for sent in X]

# Pad the sequences
X_ids = tf.keras.preprocessing.sequence.pad_sequences(X_ids, maxlen=512, truncating='post', padding='post')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_ids, y, test_size=0.2, random_state=42)

# Define the BERT model
input_ids = Input(shape=(512,), dtype=tf.int32, name='input_ids')
bert_model = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)
sequence_output = bert_model(input_ids)['last_hidden_state']
clf_output = sequence_output[:, 0, :]
out = Dense(3, activation='softmax')(clf_output)
model = Model(inputs=input_ids, outputs=out)

# Compile the model
model.compile(optimizer=Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=5, batch_size=32, callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
