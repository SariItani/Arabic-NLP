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
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from gensim.models import Word2Vec

# Load the Word2Vec model and embedding matrix
word2vec_model = Word2Vec.load("../models/dad_word2vec.model")
embedding_matrix = np.load('../models/dad_embedding_matrix.npy')

# Load the CSV file
df = pd.read_csv('../data/arabic_sentiment.csv')

# Create a dictionary to hold separate DataFrames for each emotion
emotion_dfs = {}
for column in df.columns[1:]:  # Skip the first column
    emotion_dfs[column] = df[['Unnamed: 0', column]].dropna().reset_index(drop=True)

# Function to convert text to Word2Vec feature vectors
def text_to_word2vec(texts, model, embedding_dim):
    vectors = np.zeros((len(texts), embedding_dim))
    for i, text in enumerate(texts):
        words = text.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            vectors[i] = np.mean(word_vectors, axis=0)
    return vectors

# Prepare data for each emotion using Word2Vec embeddings
word2vec_data = {}
embedding_dim = 300  # The dimension of your Word2Vec embeddings
for emotion, df_emotion in emotion_dfs.items():
    texts = df_emotion.iloc[:, 1].values
    word2vec_data[emotion] = text_to_word2vec(texts, word2vec_model, embedding_dim)

# Initialize data for performance tracking
model_performance = []

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

model_metrics = {model_name: [] for model_name, _ in models}
model_metrics["CNN"] = []  # Initialize CNN key
model_metrics["LSTM"] = []  # Initialize LSTM key

# Define deep learning models
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Train models for each emotion
for emotion in emotion_dfs.keys():
    # Get the Word2Vec matrix for the current emotion
    X = word2vec_data[emotion]
    y = np.ones(X.shape[0])  # Positive samples for the current emotion

    # Negative samples (all other emotions)
    X_negative = np.vstack([word2vec_data[emo] for emo in emotion_dfs.keys() if emo != emotion])
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

    # CNN Model
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Add a new axis for channels
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    cnn_model = build_cnn_model(input_shape=(X_train_cnn.shape[1], 1))
    cnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=32, validation_split=0.1)
    y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
    cnn_score = accuracy_score(y_test, y_pred_cnn)
    cnn_auc = roc_auc_score(y_test, cnn_model.predict(X_test_cnn))
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

    # LSTM Model
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Add a new axis for features
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = build_lstm_model(input_shape=(X_train_lstm.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train, epochs=5, batch_size=32, validation_split=0.1)
    y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
    lstm_score = accuracy_score(y_test, y_pred_lstm)
    lstm_auc = roc_auc_score(y_test, lstm_model.predict(X_test_lstm))
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

    # Save the best model for this emotion
    with open(f'../models/best_model_word2vec_{emotion}.pkl', 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save evaluation report
    with open(f'../results/evaluation_reports/word2vec_{emotion}_report.txt', 'w') as f:
        f.write(f"Classification Report for {emotion}:\n")
        f.write(classification_report(y_test, best_model.predict(X_test)))
        for model_name, score, report, conf_matrix, auc in all_model_metrics:
            f.write(f"\nModel: {model_name}\n")
            f.write(f"Accuracy: {score:.4f}\n")
            f.write(f"ROC AUC: {auc:.4f}\n")
            f.write(f"Classification Report:\n{report}\n")
    
    # Save confusion matrix
    conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {emotion}')
    plt.savefig(f'../results/figures/word2vec_{emotion}_confusion_matrix.png')
    plt.close()

    # Save ROC Curve for the best model
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {emotion} (Best Model: {type(best_model).__name__})')
    plt.savefig(f'../results/figures/word2vec_{emotion}_roc_curve.png')
    plt.close()

# Save the metrics to CSV files
for model_name, metrics in model_metrics.items():
    df_metrics = pd.DataFrame(metrics, columns=['Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])
    df_metrics.to_csv(f'../results/evaluation_reports/word2vec_{model_name}_metrics.csv', index=False)

# Convert model performance data to DataFrame
df_performance = pd.DataFrame(model_performance, columns=['Model-Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='Accuracy', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('Accuracy')
plt.ylabel('Model-Emotion')
plt.title('Model Performance vs Emotion (Word2Vec)')
plt.tight_layout()
plt.savefig('../results/figures/model_performance_vs_emotion_word2vec.png')
plt.close()

# Plot AUC scores
plt.figure(figsize=(12, 8))
sns.barplot(x='AUC', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('AUC')
plt.ylabel('Model-Emotion')
plt.title('Model AUC vs Emotion (Word2Vec)')
plt.tight_layout()
plt.savefig('../results/figures/model_auc_vs_emotion_word2vec.png')
plt.close()

# Save the performance data to a CSV file
df_performance.to_csv('../results/evaluation_reports/model_performance_word2vec.csv', index=False)

# Print the overall performance summary
print("Model Performance Summary (Word2Vec):")
print(df_performance)
