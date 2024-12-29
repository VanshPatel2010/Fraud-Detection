# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# st.title("Stock Market Prediction using Machine Learning")

# # User inputs
# symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL")
# period = st.selectbox("Select Data Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=4)
# window = st.slider("Select Window Size (days):", min_value=3, max_value=30, value=5)

# if st.button("Predict"):
#     with st.spinner("Fetching and processing data..."):
#         # Download data
#         df = yf.download(symbol, period=period)

#         if df.empty:
#             st.error(f"No data available for symbol {symbol} over the period {period}.")
#         else:
#             # Feature engineering
#             window_size = window
#             df['SMA'] = df['Close'].rolling(window=window_size).mean()

#             # Prepare data
#             X = []
#             y = []
#             for i in range(window_size, len(df)):
#                 X.append(df['Close'].iloc[i-window_size:i].values)
#                 y.append(df['Close'].iloc[i])

#             X = np.array(X)
#             y = np.array(y)
#             X = X.reshape(X.shape[0], -1)

#             # Train-test split
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#             # Train model
#             model = LinearRegression()
#             model.fit(X_train, y_train)

#             # Predict
#             y_pred = model.predict(X_test)

#             # Evaluate
#             mse = mean_squared_error(y_test, y_pred)
#             st.write(f"Mean Squared Error: {mse:.2f}")

#             # Plot results
#             fig, ax = plt.subplots(figsize=(12, 6))
#             ax.plot(df.index[-len(y_test):], y_test, label="Actual Price", color="blue")
#             ax.plot(df.index[-len(y_test):], y_pred, label="Predicted Price", color="red")
#             ax.set_title(f"{symbol} Stock Price Prediction")
#             ax.set_xlabel("Date")
#             ax.set_ylabel("Price (USD)")
#             ax.legend()
#             st.pyplot(fig)

#             # Display predictions
#             st.write("Predicted vs Actual Prices:")
#             results = pd.DataFrame({"Predicted": y_pred, "Actual": y_test})
#             st.write(results)

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import PyPDF2
import re
from datetime import datetime
import spacy
import os


class InvoiceFraudDetector:
    def __init__(self):
        try:
            # Load SpaCy model for NLP tasks
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("SpaCy model 'en_core_web_sm' not found. Please install it using:")
            print("python -m spacy download en_core_web_sm")
            raise
        # Initialize TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(max_features=100)
        # Initialize anomaly detector
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        # Historical data patterns (would be loaded from database in production)
        self.historical_patterns = {
            'avg_amount': 5000,
            'std_amount': 2000,
            'common_terms': set(['invoice', 'total', 'due date', 'payment', 'tax'])
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyPDF2."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return ""

    def extract_invoice_details(self, text):
        """Extract key details from invoice text using improved regex patterns."""
        details = {
            'amount': None,
            'date': None,
            'supplier': None,
            'terms': set(),
            'text_content': text
        }

        # Improved amount pattern to catch different formats
        amount_patterns = [
            r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Standard price
            r'Total:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Total amount
            r'Amount:?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'  # Amount field
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, text, re.IGNORECASE)
            if amount_match:
                try:
                    details['amount'] = float(amount_match.group(1).replace(',', ''))
                    break
                except:
                    continue

        # Improved date patterns
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                try:
                    date_str = date_match.group()
                    # Try multiple date formats
                    for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%B %d, %Y', '%B %d %Y']:
                        try:
                            details['date'] = datetime.strptime(date_str, fmt)
                            break
                        except:
                            continue
                    if details['date']:
                        break
                except:
                    continue

        # Use SpaCy for improved entity recognition
        doc = self.nlp(text)
        
        # Extract organization names with confidence scoring
        orgs = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['ORG', 'PERSON']]
        if orgs:
            # Prioritize organizations over person names
            org_names = [org[0] for org in orgs if org[1] == 'ORG']
            if org_names:
                details['supplier'] = org_names[0]
            elif orgs:
                details['supplier'] = orgs[0][0]

        # Extract key terms with better filtering
        details['terms'] = set([
            token.lemma_.lower() for token in doc 
            if not token.is_stop and not token.is_punct 
            and len(token.text) > 2  # Filter out very short tokens
            and token.pos_ in ['NOUN', 'VERB', 'PROPN']  # Only meaningful parts of speech
        ])

        return details

    def calculate_risk_scores(self, invoice_details):
        """Calculate risk scores with improved anomaly detection."""
        risk_factors = {
            'amount_anomaly': 0,
            'missing_info': 0,
            'term_anomaly': 0,
            'supplier_risk': 0,
            'date_anomaly': 0
        }

        # Enhanced amount anomaly detection
        if invoice_details['amount']:
            z_score = (invoice_details['amount'] - self.historical_patterns['avg_amount']) / self.historical_patterns['std_amount']
            risk_factors['amount_anomaly'] = min(abs(z_score) / 3, 1)
            
            # Additional check for round numbers (potential fake invoices)
            if invoice_details['amount'] % 100 == 0:
                risk_factors['amount_anomaly'] += 0.2
        else:
            risk_factors['missing_info'] += 0.5

        # Improved missing information scoring
        if not invoice_details['date']:
            risk_factors['missing_info'] += 0.3
        if not invoice_details['supplier']:
            risk_factors['missing_info'] += 0.2

        # Date anomaly detection
        if invoice_details['date']:
            days_old = (datetime.now() - invoice_details['date']).days
            if days_old > 180:  # Suspicious if invoice is too old
                risk_factors['date_anomaly'] = min(days_old / 365, 1)

        # Enhanced term analysis
        required_terms = {'invoice', 'total', 'payment', 'due'}
        found_terms = invoice_details['terms'].intersection(required_terms)
        risk_factors['term_anomaly'] = 1 - (len(found_terms) / len(required_terms))

        # Updated weights for risk factors
        weights = {
            'amount_anomaly': 0.35,
            'missing_info': 0.25,
            'term_anomaly': 0.15,
            'supplier_risk': 0.1,
            'date_anomaly': 0.15
        }

        final_score = sum(score * weights[factor] for factor, score in risk_factors.items())
        return final_score, risk_factors

    def analyze_invoices(self, invoice_directory):
        """Analyze a batch of invoices and return risk assessments."""
        results = []

        for filename in os.listdir(invoice_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(invoice_directory, filename)
                
                # Extract and analyze invoice
                text = self.extract_text_from_pdf(file_path)
                details = self.extract_invoice_details(text)
                risk_score, risk_factors = self.calculate_risk_scores(details)

                # Prepare result
                result = {
                    'filename': filename,
                    'risk_score': risk_score,
                    'risk_factors': risk_factors,
                    'details': details
                }
                results.append(result)

        # Sort results by risk score
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        return results

    def generate_report(self, results):
        """Generate a detailed report of fraud analysis results."""
        report = "Invoice Fraud Detection Report\n"
        report += "=" * 30 + "\n\n"

        for result in results:
            report += f"File: {result['filename']}\n"
            report += f"Risk Score: {result['risk_score']:.2f}\n"
            report += "Risk Factors:\n"
            for factor, score in result['risk_factors'].items():
                report += f"  - {factor}: {score:.2f}\n"
            report += "\nDetails:\n"
            report += f"  Amount: ${result['details']['amount']:.2f}\n" if result['details']['amount'] else "  Amount: Not found\n"
            report += f"  Date: {result['details']['date']}\n" if result['details']['date'] else "  Date: Not found\n"
            report += f"  Supplier: {result['details']['supplier']}\n" if result['details']['supplier'] else "  Supplier: Not found\n"
            report += "\n" + "-" * 30 + "\n"

        return report

def main():
    # Initialize detector
    detector = InvoiceFraudDetector()
    
    # Set path to the invoices directory
    invoice_directory = "invoices"
    
    try:
        # Analyze invoices
        results = detector.analyze_invoices(invoice_directory)
        
        # Generate and print report
        report = detector.generate_report(results)
        print(report)
        
        # Save high-risk invoices to file
        with open('fraud_detection_report.txt', 'w') as f:
            f.write(report)
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()