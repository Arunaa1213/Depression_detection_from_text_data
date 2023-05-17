import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the preprocessed data from CSV file
data = pd.read_csv('preprocessed_data.csv')

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000)

# Fit the vectorizer on the text data and transform the data into a sparse matrix
tfidf_data = tfidf_vectorizer.fit_transform(data['Text_data'])

# Convert the sparse matrix to a DataFrame
tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_data, columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate the TF-IDF DataFrame with the PID and Label columns
preprocessed_data = pd.concat([data[['PID', 'Label']], tfidf_df], axis=1)

# Save the preprocessed data to CSV file
preprocessed_data.to_csv('tf_idf_data.csv', index=False)
