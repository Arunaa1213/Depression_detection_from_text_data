import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the dataset into a pandas DataFrame
df = pd.read_csv('dataset_with_ngrams.csv')
test_df = pd.read_csv('preprocessed_test_data.csv')
# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Fit the vectorizer on the text data
vectorizer.fit(df['Text_data'])

# Transform the n-grams into TF-IDF vectors
X = vectorizer.transform(df['Text_data'])
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model on the training data
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)






# Evaluate the performance of the tuned model on the testing data
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


test_df = test_df.dropna(subset=['Text_data'])
test_df['Text_data'] = test_df['Text_data'].fillna(' ')

X_test = vectorizer.transform(test_df['Text_data'])
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model on the test data
# accuracy = accuracy_score(y_test, y_pred)


test_df['Predicted_Label'] = y_pred
test_df.to_csv('test_predictions.csv', index=False)