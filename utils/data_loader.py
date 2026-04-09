import pandas as pd

def load_data(file_path):
    df = pd.read_csv("C:/Users/aditi/Desktop/Mtech AI Cummins/Sem 2/Deep Learning/DL Project/train.csv")
    return df

def dataframe_to_text(df):

    rows = []

    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        rows.append(text)

    return rows