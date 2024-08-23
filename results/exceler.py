import pandas as pd

# Updated Data
accuracy_data = {
    "Metric": ["Overall Accuracy"],
    "Averaging Accuracy": [0.9643366619115549],
    "Voting Accuracy": [0.9586305278174037]
}

classification_report_data = {
    "Emotion": ["شعور البغض - Hatred", "شعور الحب - Liking", "شعور الحزن – Sadness", "شعور الخوف – Fear", "شعور الفرح - Joy"],
    "Precision (Averaging)": [0.98, 0.97, 0.94, 0.95, 0.99],
    "Recall (Averaging)": [0.96, 1.00, 0.95, 0.97, 0.94],
    "F1-Score (Averaging)": [0.97, 0.98, 0.94, 0.96, 0.96],
    "Support (Averaging)": [126, 150, 150, 150, 125],
    "Precision (Voting)": [0.92, 0.97, 0.93, 0.99, 0.99],
    "Recall (Voting)": [0.99, 1.00, 0.95, 0.97, 0.87],
    "F1-Score (Voting)": [0.95, 0.98, 0.94, 0.98, 0.93],
    "Support (Voting)": [126, 150, 150, 150, 125]
}

# Create DataFrames
df_accuracy = pd.DataFrame(accuracy_data)
df_classification_report = pd.DataFrame(classification_report_data)

# Write to Excel
with pd.ExcelWriter('./ensemble_model_results.xlsx', engine='openpyxl') as writer:
    df_accuracy.to_excel(writer, sheet_name='Accuracy', index=False)
    df_classification_report.to_excel(writer, sheet_name='Classification Report', index=False)
