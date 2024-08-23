import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create directories for saving results
os.makedirs("../results/evaluation_reports", exist_ok=True)
os.makedirs("../results/figures", exist_ok=True)

# Load preprocessed data and labels
emotions = [
    "شعور البغض - Hatred",
    "شعور الحب - Liking",
    "شعور الحزن – Sadness ", 
    "شعور الخوف – Fear",
    "شعور الفرح - Joy"
]
data = {emotion: np.load(f'../data/data_{emotion}.npy') for emotion in emotions}
labels = {emotion: np.load(f'../data/labels_{emotion}.npy') for emotion in emotions}

# Load tokenizer and embedding matrix
with open('../models/dad_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
embedding_matrix = np.load('../models/dad_embedding_matrix.npy')
maxlen = 100
embedding_dim = 300

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

# Initialize a list to store the best model performances for plotting
best_model_performance = []

# Helper function to evaluate models
def evaluate_model(model, model_name, X_test, y_test, emotion):
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    score = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = report.get('0.0', {})
    f1 = metrics.get('f1-score', None)
    recall = metrics.get('recall', None)
    precision = metrics.get('precision', None)

    # Save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+emotion, emotion], yticklabels=['Not '+emotion, emotion])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {emotion} - {model_name}')
    plt.savefig(f'../results/figures/{model_name}_{emotion}_confusion_matrix.png')
    plt.close()
    
    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {emotion} (Model: {model_name})')
    plt.savefig(f'../results/figures/{model_name}_{emotion}_roc_curve.png')
    plt.close()

    return score, auc, f1, recall, precision

# Train models for each emotion
for emotion in emotions:
    X_positive = data[emotion]
    y_positive = np.ones(X_positive.shape[0])

    # Negative samples (all other emotions)
    X_negative = np.concatenate([data[emo] for emo in emotions if emo != emotion])
    y_negative = np.zeros(X_negative.shape[0])

    # Create a balanced dataset by combining positive and negative samples
    X_combined = np.vstack([X_positive, X_negative])
    y_combined = np.concatenate([y_positive, y_negative])

    # Shuffle combined dataset
    indices = np.arange(X_combined.shape[0])
    np.random.shuffle(indices)
    X_combined, y_combined = X_combined[indices], y_combined[indices]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    best_score = 0
    best_model = None
    best_model_name = None

    # Evaluate traditional models
    for model_name, model in models:
        try:
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            score, auc, f1, recall, precision = evaluate_model(model, model_name, X_test, y_test, emotion)
            model_metrics[model_name].append((emotion, score, auc, f1, recall, precision))
            if score > best_score:
                best_score = score
                best_model = model
                best_auc = auc
                best_f1, best_recall, best_precision = f1, recall, precision
                best_model_name = model_name
        except ValueError as e:
            print(f"Skipping {model_name} for {emotion}: {e}")

    # Save the best traditional model
    with open(f'../models/best_model_{emotion}.pkl', 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate CNN model
    cnn_model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    X_test_padded = pad_sequences(X_test, maxlen=maxlen)
    score, auc, f1, recall, precision = evaluate_model(cnn_model, "CNN", X_test_padded, y_test, emotion)
    model_metrics["CNN"].append((emotion, score, auc, f1, recall, precision))
    if score > best_score:
        best_score = score
        best_model = cnn_model
        best_auc = auc
        best_f1, best_recall, best_precision = f1, recall, precision
        best_model_name = "CNN"

    # Save CNN model if it's the best
    if best_model_name == "CNN":
        best_model.save(f'../models/best_model_cnn_{emotion}.h5')

    # Evaluate LSTM model
    lstm_model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    score, auc, f1, recall, precision = evaluate_model(lstm_model, "LSTM", X_test_padded, y_test, emotion)
    model_metrics["LSTM"].append((emotion, score, auc, f1, recall, precision))
    if score > best_score:
        best_score = score
        best_model = lstm_model
        best_auc = auc
        best_f1, best_recall, best_precision = f1, recall, precision
        best_model_name = "LSTM"

    # Save LSTM model if it's the best
    if best_model_name == "LSTM":
        best_model.save(f'../models/best_model_lstm_{emotion}.h5')

    # Track the best model's performance
    best_model_performance.append((f"{best_model_name} - {emotion}", best_score, best_auc, best_f1, best_recall, best_precision))

# Save the metrics to CSV files
for model_name, metrics in model_metrics.items():
    df_metrics = pd.DataFrame(metrics, columns=['Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])
    df_metrics.to_csv(f'../results/evaluation_reports/_{model_name}_metrics.csv', index=False)

# Save the best model metrics to CSV files
df_performance = pd.DataFrame(best_model_performance, columns=['Model-Emotion', 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision'])
df_performance.to_csv('../results/evaluation_reports/best_model_performance.csv', index=False)

# Plotting Accuracy of the best models
plt.figure(figsize=(12, 8))
sns.barplot(x='Accuracy', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('Accuracy')
plt.ylabel('Model-Emotion')
plt.title('Best Model Performance vs Emotion')
plt.tight_layout()
plt.savefig('../results/figures/best_model_performance_vs_emotion.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(x='AUC', y='Model-Emotion', data=df_performance, palette='viridis')
plt.xlabel('AUC')
plt.ylabel('Model-Emotion')
plt.title('Model AUC vs Emotion')
plt.tight_layout()
plt.savefig('../results/figures/model_auc_vs_emotion.png')
plt.close()

print("Evaluation and model saving completed.")

# Print the overall performance summary
print("Model Performance Summary:")
print(df_performance)