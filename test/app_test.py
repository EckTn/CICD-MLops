import skops.io as sio
import pandas as pd
import pytest

pipeline = sio.load("../model/loan_pipeline.skops", trusted=True)

def test_predict_loan():
    # Sample input features
    input_features = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 6875,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 200,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Semiurban"
    }
    total_income = input_features["ApplicantIncome"] + input_features["CoapplicantIncome"]
    emi = input_features["LoanAmount"] / input_features["Loan_Amount_Term"] if input_features["Loan_Amount_Term"] != 0 else 0

    # Create DataFrame for the input features
    features = pd.DataFrame([list(input_features.values()) + [total_income, emi]], columns=list(input_features.keys()) + ['TotalIncome', 'EMI'])

    try:
        # Make predictions using the loaded pipeline
        predicted_decision = pipeline.predict(features)[0]
        expected_decision = 1

        assert predicted_decision == expected_decision, f"Test failed: Expected {expected_decision}, got {predicted_decision}"
        print(f"Loan Approval Decision: {'Approved' if predicted_decision == 1 else 'Not Approved'}")
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")

test_predict_loan()
