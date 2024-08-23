import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
import seaborn as sns
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create directories for saving results
os.makedirs("../results/evaluation_reports", exist_ok=True)
os.makedirs("../results/figures", exist_ok=True)

# Load the CSV file
df = pd.read_csv('../data/arabic_sentiment.csv')

# Create a dictionary to hold separate DataFrames for each emotion
emotion_dfs = {}

# Create separate DataFrames for each emotion column
for column in df.columns[1:]:  # Skip the first column
    emotion_dfs[column] = df[['Unnamed: 0', column]].dropna().reset_index(drop=True)

# Load pre-trained Bag of Words vectorizer
with open('../models/bow_vectorizer.pkl', 'rb') as handle:
    vectorizer = pickle.load(handle)

# Transform each emotion DataFrame using the fitted vectorizer
bow_data = {}
for emotion, df_emotion in emotion_dfs.items():
    texts = df_emotion.iloc[:, 1].values
    bow_matrix = vectorizer.transform(texts)
    bow_data[emotion] = bow_matrix

# Load tokenizer and embedding matrix for deep learning models
with open('../models/dad_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
embedding_matrix = np.load('../models/dad_embedding_matrix.npy')
maxlen = 100
embedding_dim = 300

# Prepare data for deep learning models
tokenized_data = {}
for emotion, df_emotion in emotion_dfs.items():
    texts = df_emotion.iloc[:, 1].values
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    tokenized_data[emotion] = padded_sequences

# Define models to test
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC(probability=True)),
    ('MLP', MLPClassifier(max_iter=1000)),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Bernoulli Naive Bayes', BernoulliNB())
]

# Initialize dictionaries to store performance metrics for each model
model_metrics = {model_name: [] for model_name, _ in models}
model_metrics["CNN"] = []  # Initialize CNN key
model_metrics["LSTM"] = []  # Initialize LSTM key

model_performance = []

# Train models for each emotion
for emotion in emotion_dfs.keys():
    X = bow_data[emotion].toarray()
    y = np.ones(X.shape[0])  # Positive samples for the current emotion
    
    # Negative samples (all other emotions)
    X_negative = np.vstack([bow_data[emo].toarray() for emo in emotion_dfs.keys() if emo != emotion])
    y_negative = np.zeros(X_negative.shape[0])
    
    # Create a balanced dataset by combining positive and negative samples
    X_combined = np.vstack([X, X_negative])
    y_combined = np.concatenate([y, y_negative])
    
    # Shuffle combined dataset
    indices = np.arange(X_combined.shape[0])
    np.random.shuffle(indices)
    X_combined, y_combined = X_combined[indices], y_combined[indices]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
    
    best_score = 0
    best_auc = 0
    best_model = None
    best_y_pred_prob = None
    all_model_metrics = []

    for model_name, model in models:
        try:
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            score = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics = report.get('0.0', {})
            f1 = metrics.get('f1-score', None)
            recall = metrics.get('recall', None)
            precision = metrics.get('precision', None)
            
            # Save accuracy and ROC AUC for each model-emotion pair
            model_metrics[model_name].append((emotion, score, auc, f1, recall, precision))

            # Save confusion matrix and ROC Curve as before
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10,7))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {emotion} - {model_name}')
            plt.savefig(f'../results/figures/{model_name}_{emotion}_confusion_matrix.png')
            plt.close()
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, marker='.')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {emotion} (Model: {model_name})')
            plt.savefig(f'../results/figures/{model_name}_{emotion}_roc_curve.png')
            plt.close()

            if score > best_score:
                best_score = score
                best_auc = auc
                best_f1 = f1
                best_recall = recall
                best_precision = precision
                best_model = model
                best_y_pred_prob = y_pred_prob

        except ValueError as e:
            print(f"Skipping {model_name} for {emotion}: {e}")
    
    # Save the best model for this emotion
    with open(f'../models/best_model_bow_{emotion}.pkl', 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # CNN model
    cnn_model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_train_padded = pad_sequences(X_train, maxlen=maxlen)
    X_test_padded = pad_sequences(X_test, maxlen=maxlen)
    cnn_model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.1)
    y_pred_prob = cnn_model.predict(X_test_padded).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    score = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = report.get('0.0', {})
    f1 = metrics.get('f1-score', None)
    recall = metrics.get('recall', None)
    precision = metrics.get('precision', None)
    
    # Save CNN accuracy and ROC AUC
    model_metrics["CNN"].append((emotion, score, auc, f1, recall, precision))

    # Save CNN confusion matrix and ROC Curve as before
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {emotion} - CNN')
    plt.savefig(f'../results/figures/CNN_{emotion}_confusion_matrix.png')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {emotion} (Model: CNN)')
    plt.savefig(f'../results/figures/CNN_{emotion}_roc_curve.png')
    plt.close()
    
    if score > best_score:
        best_score = score
        best_auc = auc
        best_f1 = f1
        best_recall = recall
        best_precision = precision
        best_model = ("CNN", cnn_model)

    # LSTM model
    lstm_model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.1)
    y_pred_prob = lstm_model.predict(X_test_padded).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    score = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = report.get('0.0', {})
    f1 = metrics.get('f1-score', None)
    recall = metrics.get('recall', None)
    precision = metrics.get('precision', None)
    
    # Save LSTM accuracy and ROC AUC
    model_metrics["LSTM"].append((emotion, score, auc, f1, recall, precision))

    # Save LSTM confusion matrix and ROC Curve as before
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {emotion} - LSTM')
    plt.savefig(f'../results/figures/LSTM_{emotion}_confusion_matrix.png')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {emotion} (Model: LSTM)')
    plt.savefig(f'../results/figures/LSTM_{emotion}_roc_curve.png')
    plt.close()
    
    if score > best_score:
        best_score = score
        best_auc = auc
        best_f1 = f1
        best_recall = recall
        best_precision = precision
        best_model = ("LSTM", lstm_model)

    # Store the best model and its performance
    model_performance.append((f"{type(best_model).__name__} - {emotion}", best_score, best_auc, best_f1, best_recall, best_precision))

# Save the metrics to CSV files
for model_name, metrics in model_metrics.items():
    df_metrics = pd.DataFrame(metrics, columns=['Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])
    df_metrics.to_csv(f'../results/evaluation_reports/bow_{model_name}_metrics.csv', index=False)

# Convert best model performance data to DataFrame
df_performance = pd.DataFrame(model_performance, columns=['Model-Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])

# Plotting Accuracy
plt.figure(figsize=(12, 8))
sns.barplot(x='Accuracy', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('Accuracy')
plt.ylabel('Model-Emotion')
plt.title('Model Performance vs Emotion (Bag of Words)')
plt.tight_layout()
plt.savefig('../results/figures/model_performance_vs_emotion_bow.png')
plt.close()

# Plot AUC scores
plt.figure(figsize=(12, 8))
sns.barplot(x='AUC', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('AUC')
plt.ylabel('Model-Emotion')
plt.title('Model AUC Scores vs Emotion (Bag of Words)')
plt.tight_layout()
plt.savefig('../results/figures/model_auc_vs_emotion_bow.png')
plt.close()

# Save the performance data to a CSV file
df_performance.to_csv('../results/evaluation_reports/model_performance_bow.csv', index=False)

# Print the overall performance summary
print("Model Performance Summary (Bag of Words):")
print(df_performance)
