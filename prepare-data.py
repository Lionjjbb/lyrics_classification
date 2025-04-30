# prepare_data.py

import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources (only need to run once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk.download('omw-1.4')  # For lemmatizer support

# Path to the combined dataset
input_path = 'data/lyrics_top3.csv'  # Update the path if necessary

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
word_list = set(words.words())  # English words corpus
lemmatizer = WordNetLemmatizer()

# Add custom stop words
custom_stop_words = {'chorus', 'verse', 'repeat', 'bridge', 'instrumental', 'intro', 'outro',
                     'yeah', 'oh', 'na', 'la', 'hey', 'ho', 'yo', 'baby'}
stop_words.update(custom_stop_words)

# Function to remove elongated words (e.g., soooo to soo)
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

# Comprehensive contractions dictionary
contractions_dict = {
    "won't": "will not",
    "can't": "cannot",
    # ... (include more contractions as needed)
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "n't": " not",
    "'ve": " have",
    "'m": " am",
}

def expand_contractions(text, contractions_dict=contractions_dict):
    pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), flags=re.IGNORECASE)
    def replace(match):
        return contractions_dict[match.group(0).lower()]
    return pattern.sub(replace, text)

# Function to clean text
def clean_text(text):
    # Check for NaN
    if not isinstance(text, str):
        return ''
    # Lowercase
    text = text.lower()
    # Remove text inside brackets
    text = re.sub(r'\[.*?\]', '', text)
    # Expand contractions
    text = expand_contractions(text)
    # Remove non-alphabetic characters and extra spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove elongated words
    text = reduce_lengthening(text)
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]
    # Remove non-English words
    words = [word for word in words if word in word_list]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text

# Read the data
print("Reading data...")
data_df = pd.read_csv(input_path)

# Drop missing values
data_df.dropna(subset=['tag', 'lyrics'], inplace=True)

# Clean the lyrics
print("Cleaning lyrics...")
data_df['lyrics'] = data_df['lyrics'].apply(clean_text)

# Remove empty lyrics after cleaning
data_df = data_df[data_df['lyrics'].str.strip() != '']

# Create tag to label mapping
print("Creating tag to label mapping...")
data_df['tag'] = data_df['tag'].str.lower()
all_tags = data_df['tag'].unique()
tag_to_label = {tag: idx for idx, tag in enumerate(all_tags)}
label_to_tag = {idx: tag for tag, idx in tag_to_label.items()}

# Map tags to numerical labels
data_df['label'] = data_df['tag'].map(tag_to_label)

# Save tag to label mapping for reference
with open('tag_to_label_mapping.txt', 'w') as f:
    for tag, label in tag_to_label.items():
        f.write(f"{label}: {tag}\n")

# Prepare corpus and labels
corpus = data_df['lyrics'].tolist()
labels = data_df['label'].values

# Fit the TF-IDF vectorizer
print("Fitting TF-IDF vectorizer...")
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=20000,   # Reduced number of features
    ngram_range=(1,2),    # Unigrams and bigrams
    min_df=100,             # Ignore terms that appear in less than 5 documents
    max_df=0.6            # Ignore terms that appear in more than 60% of documents
)
X = vectorizer.fit_transform(corpus)
y = labels

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# Feature selection using chi-squared test
print("Performing feature selection...")
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=10000)  # Reduced number of features
X_selected = selector.fit_transform(X_scaled, y)

# Save processed data
print("Saving processed data...")
from scipy import sparse
sparse.save_npz('data/X_top3.npz', X_selected)
pd.DataFrame({'label': y}).to_csv('data/y_top3.csv', index=False)

# Save the vectorizer, scaler, and selector
import pickle
with open('vectorizer_top3.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('scaler_top3.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('selector_top3.pkl', 'wb') as f:
    pickle.dump(selector, f)

print("Data preparation completed successfully.")
