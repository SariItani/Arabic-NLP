import pandas as pd

# Define the path to the Excel file
excel_path = "./baba copy.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_path)

# Define the path for the output CSV file
csv_path = "./arabic_sentiment.csv"

# Save the DataFrame as a CSV file with an index
df.to_csv(csv_path, index=True)
