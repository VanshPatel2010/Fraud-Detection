import streamlit as st
import os
from test import InvoiceFraudDetector  # Assuming your class is saved in a separate file

# Initialize detector
detector = InvoiceFraudDetector()

st.title("Invoice Fraud Detection")
st.write("Upload PDF invoices to analyze their fraud risk.")

# Upload files
uploaded_files = st.file_uploader("Choose PDF invoices", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Analyze the uploaded file
        text = detector.extract_text_from_pdf(temp_file_path)
        details = detector.extract_invoice_details(text)
        risk_score, risk_factors = detector.calculate_risk_scores(details)

        # Prepare the result
        result = {
            "filename": uploaded_file.name,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "details": details,
        }
        results.append(result)

    # Display results
    st.subheader("Analysis Results")
    for result in results:
        st.write(f"**File:** {result['filename']}")
        st.write(f"**Risk Score:** {result['risk_score']:.2f}")
        st.write("**Risk Factors:**")
        for factor, score in result["risk_factors"].items():
            st.write(f"  - {factor}: {score:.2f}")
        st.write("**Details:**")
        st.write(f"  Amount: ${result['details']['amount']:.2f}" if result["details"]["amount"] else "  Amount: Not found")
        st.write(f"  Date: {result['details']['date']}" if result["details"]["date"] else "  Date: Not found")
        st.write(f"  Supplier: {result['details']['supplier']}" if result["details"]["supplier"] else "  Supplier: Not found")
        st.write("---")
    
    # Optionally save the report
    st.download_button("Download Results as Text", detector.generate_report(results), file_name="fraud_report.txt")

# Cleanup temporary files
if os.path.exists("temp"):
    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))
    os.rmdir("temp")
