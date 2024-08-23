import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from stop_words import get_stop_words
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools

# Load the preprocessed data
preprocessed_data = pd.read_csv('../data/arabic_sentiment.csv')

# Load the processed data
with open('../models/dad_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Download NLTK stop words
nltk.download('stopwords')
arabic_stop_words = set(nltk.corpus.stopwords.words('arabic'))
additional_stop_words = set(get_stop_words('ar'))
all_stop_words = arabic_stop_words.union(additional_stop_words)

# Define a function to normalize and remove stop words from text
def preprocess_text(text):
    # Normalize Arabic text
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[ًٌٍَُِْ]", "", text)
    
    # Remove stop words
    return ' '.join([word for word in text.split() if word not in all_stop_words])

# Define a function to plot word clouds
def plot_wordcloud(text, title):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    wordcloud = WordCloud(font_path='DejaVuSans.ttf', background_color='white', width=800, height=400).generate(bidi_text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"../results/figures/{title}.png")

# Define a function to plot token distribution
def plot_token_distribution(tokenized_data, title):
    token_lengths = [len(tokens) for tokens in tokenized_data]
    sns.histplot(token_lengths, kde=True)
    plt.title(title)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.savefig(f"../results/figures/{title}.png")

# Define a function to plot most common words
def plot_most_common_words(text, title, num_words=20):
    words = text.split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(num_words)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=list(counts), y=list(words))
    plt.title(title)
    plt.xlabel('Counts')
    plt.ylabel('Words')
    plt.savefig(f"../results/figures/{title}.png")

# Define a function to plot feature distributions
def plot_feature_distribution(df, title):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if numerical_cols.size == 0:
        print(f"No numerical columns to plot in {title}")
        return
    
    plt.figure(figsize=(10, 6))
    df[numerical_cols].hist(bins=50, figsize=(20, 15))
    plt.suptitle(title)
    plt.savefig(f"../results/figures/{title}.png")

# EDA on preprocessed data
print("EDA on Preprocessed Data")

# Plot word cloud for each emotion
for column in preprocessed_data.columns[1:]:
    text = ' '.join(preprocessed_data[column].dropna().apply(preprocess_text))
    plot_wordcloud(text, f'Word Cloud for {column}')

# Plot most common words for each emotion
for column in preprocessed_data.columns[1:]:
    text = ' '.join(preprocessed_data[column].dropna().apply(preprocess_text))
    plot_most_common_words(text, f'Most Common Words for {column}')

# Plot feature distribution
preprocessed_data_features = preprocessed_data[preprocessed_data.columns[1:]]
plot_feature_distribution(preprocessed_data_features, 'Feature Distribution for Preprocessed Data')

# EDA on processed data
print("EDA on Processed Data")

# Load processed data
tokenized_data = {}
maxlen = 100
for column in preprocessed_data.columns[1:]:
    texts = preprocessed_data[column].dropna().apply(preprocess_text).values
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    tokenized_data[column] = padded_sequences

# Plot token distribution for each emotion
for emotion, data in tokenized_data.items():
    plot_token_distribution(data, f'Token Distribution for {emotion}')

# Example of how to access word index from tokenizer
word_index = tokenizer.word_index
print(f"Number of unique tokens: {len(word_index)}")

# Example to plot the frequency of top N tokens
def plot_token_frequency(word_index, title, num_words=20):
    word_freq = sorted(word_index.items(), key=lambda x: x[1], reverse=True)[:num_words]
    words, counts = zip(*word_freq)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=list(counts), y=list(words))
    plt.title(title)
    plt.xlabel('Counts')
    plt.ylabel('Words')
    plt.savefig(f"../results/figures/{title}.png")

plot_token_frequency(word_index, 'Top 20 Tokens by Frequency')

print("EDA complete.")
