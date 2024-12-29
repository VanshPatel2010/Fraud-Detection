# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Parse arguments
# parser = argparse.ArgumentParser(description='Stock Market Prediction using ML')
# parser.add_argument('--symbol', type=str, required=True, help='Symbol of Stock to use')
# parser.add_argument('--period', type=str, default="2y", help='Data period to download (e.g., 2y, 1mo, etc.)')
# parser.add_argument('--window', type=int, default=5, help='Window size (number of previous days to use for prediction)')
# args = parser.parse_args()

# # Download data
# print(f"Downloading data for {args.symbol}...")
# df = yf.download(args.symbol, period=args.period)

# if df.empty:
#     raise ValueError(f"No data available for symbol {args.symbol} over the period {args.period}.")

# # Feature Engineering: Use the past `window` days to predict the next day's price
# window_size = args.window
# df['SMA'] = df['Close'].rolling(window=window_size).mean()  # Adding Simple Moving Average (SMA) as a feature

# # Prepare data for training
# X = []
# y = []

# # Iterate from `window_size` to the end of the DataFrame
# for i in range(window_size, len(df)):
#     X.append(df['Close'].iloc[i-window_size:i].values)  # Past `window_size` days' closing prices as features
#     y.append(df['Close'].iloc[i])  # Next day's closing price as target

# X = np.array(X)
# y = np.array(y)

# # Reshape X to 2D (samples, features), where features is the window size
# X = X.reshape(X.shape[0], -1)  # Flatten the 3D array to 2D (n_samples, n_features)

# # Split the data into training and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Train a Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# # Plot the predicted vs actual prices
# plt.figure(figsize=(12, 6))
# plt.plot(df.index[-len(y_test):], y_test, label='Actual Price', color='blue')
# plt.plot(df.index[-len(y_test):], y_pred, label='Predicted Price', color='red')
# plt.title(f'{args.symbol} Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.show()

# # Show some predictions
# print(f"Predicted vs Actual prices for the last few days:")
# for i in range(len(y_pred)):
#     print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test[i]:.2f}")

import streamlit as st
from test import InvoiceFraudDetector  # Assuming your class is in another file

def main():
    st.title("Invoice Fraud Detector")
    st.subheader("Upload your invoice PDFs for analysis")

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        detector = InvoiceFraudDetector()
        results = []

        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, 'wb') as f:
                f.write(uploaded_file.read())
            
            text = detector.extract_text_from_pdf(uploaded_file.name)
            details = detector.extract_invoice_details(text)
            risk_score, risk_factors = detector.calculate_risk_scores(details)
            
            results.append({
                'filename': uploaded_file.name,
                'risk_score': risk_score,
                'details': details
            })

        st.subheader("Analysis Results")
        for result in results:
            st.write(f"File: {result['filename']}")
            st.write(f"Risk Score: {result['risk_score']:.2f}")
            st.write("Details:", result['details'])

if __name__ == "__main__":
    main()
