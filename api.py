from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
from pylab import rcParams
import pickle

rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

def predict(path,X_new):
    with open(path, "rb") as file:
        model = pickle.load(file)
    expected_features = model.feature_names_in_

    # Ensure X_new has the same features as expected
    for col in expected_features:
        if col not in X_new.columns:
            X_new[col] = 0  # Add missing features with default value

    # Ensure the column order matches
    X_new = X_new[expected_features]

    # Make predictions
    predictions = model.predict(X_new)
    return predictions

def apply_transformations(df, filename="transformations.pkl"):
    with open(filename, "rb") as f:
        transformations = pickle.load(f)
    
    # Convert to numeric where needed
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
    
    # Apply Count Encoding
    df["payer_email_transaction_count"] = df["payer_email_anonymous"].map(transformations["payer_email_transaction_count"]).fillna(0)
    df["payer_mobile_transaction_count"] = df["payer_mobile_anonymous"].map(transformations["payer_mobile_transaction_count"]).fillna(0)
    df["unique_payers_per_payee"] = df["payee_id_anonymous"].map(transformations["unique_payers_per_payee"]).fillna(0)
    df["payee_ip_transaction_count"] = df["payee_ip_anonymous"].map(transformations["payee_ip_transaction_count"]).fillna(0)
    
    # Apply Frequency Encoding
    df["payer_browser_frequency"] = df["payer_browser_anonymous"].map(transformations["browser_counts"]).fillna(0)
    
    # Apply Mean Transaction per Payee
    df["payee_avg_transaction_amount"] = df["payee_id_anonymous"].map(transformations["payee_avg_transaction_amount"]).fillna(df["transaction_amount"].mean())
    
    # Apply Frequency Encoding for Payment Gateway Bank
    df["payment_gateway_bank_freq"] = df["payment_gateway_bank_anonymous"].map(transformations["payment_gateway_bank_freq"]).fillna(0)
    
    # Apply One-Hot Encoding
    df = pd.get_dummies(df, columns=["transaction_channel", "transaction_payment_mode_anonymous"], prefix=["channel", "payment_mode"], dtype=int)
    
    # Ensure one-hot encoded columns exist in new data
    missing_cols = set(transformations["one_hot_columns"]) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Add missing columns with default value 0

    df.fillna(df.median(numeric_only=True), inplace=True)


    df.drop(
        columns=[
            "transaction_date", 
            "payment_gateway_bank_anonymous", 
            "payer_browser_anonymous", 
            "payer_email_anonymous", 
            "payee_ip_anonymous", 
            "payer_mobile_anonymous", 
            "payee_id_anonymous",
            "transaction_id_anonymous"
        ], 
        inplace=True
    )
    

    columns = df.columns.tolist()
    # Filter the columns to remove data we do not want 
    columns = [c for c in columns if c not in ["Class"]]
    # Store the variable we are predicting 
    target = "is_fraud"
    # Define a random state 
    state = np.random.RandomState(42)
    X_new = df[columns]

    return X_new

app = FastAPI(title="TransactAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
async def ml_predict(api_data: dict = Body(...)):
    mapped_data = [{
        "transaction_amount": api_data.get("transaction_amount", 0),
        "transaction_date": api_data.get("transaction_date", ""),
        "transaction_channel": api_data.get("transaction_channel", ""),
        "transaction_payment_mode_anonymous": api_data.get("transaction_payment_mode", ""),
        "payment_gateway_bank_anonymous": api_data.get("payment_gateway_bank", ""),
        "payer_browser_anonymous": api_data.get("payer_browser", ""),
        "payer_email_anonymous": api_data.get("payer_email", ""),
        "payee_ip_anonymous": api_data.get("payee_ip", ""), 
        "payer_mobile_anonymous": api_data.get("payer_mobile", ""),
        "transaction_id_anonymous": api_data.get("transaction_id", ""),
        "payee_id_anonymous": api_data.get("payee_id", "")
    }]


    df = pd.DataFrame(mapped_data)
    csv_filename = "./transactions.csv"
    df.to_csv(csv_filename, index=False)
    test_df=pd.read_csv('./transactions.csv')
    X_new ,Y_new= apply_transformations(test_df)
    prediction = predict('best_model.pkl',X_new)

    return {"prediction":prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
