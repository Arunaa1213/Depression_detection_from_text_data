import pandas as pd
import re
import string

# Load the data from CSV file
data = pd.read_csv('./data/train_data.csv')

# Remove unnecessary columns
# data = data.drop(['PID'], axis=1)

# Convert text to lowercase
data['Text_data'] = data['Text_data'].str.lower()

# Remove numbers and special characters
data['Text_data'] = data['Text_data'].apply(lambda x: re.sub(r'\d+', '', x))
data['Text_data'] = data['Text_data'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

# Tokenize the text
data['Text_data'] = data['Text_data'].apply(lambda x: x.split())

# Remove stop words
stop_words = set(['a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'])
data['Text_data'] = data['Text_data'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatize the words
lemmatizer = {'is': 'be', 'are': 'be', 'am': 'be', 'was': 'be', 'were': 'be', 'has': 'have', 'have': 'have', 'had': 'have', 'get': 'get', 'got': 'get', 'gotten': 'get', 'go': 'go', 'went': 'go', 'goes': 'go', 'do': 'do', 'did': 'do', 'does': 'do', 'make': 'make', 'made': 'make', 'makes': 'make'}
data['Text_data'] = data['Text_data'].apply(lambda x: [lemmatizer.get(word, word) for word in x])

# Convert the labels to numeric values
data['Label'] = data['Label'].map({'severe': 2, 'moderate': 1, 'not depression': 0})

# Save the preprocessed data to CSV file
data.to_csv('preprocessed_data.csv', index=False)
