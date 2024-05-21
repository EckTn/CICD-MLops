import pandas as pd
import gradio as gr
import skops.io as sio

# Load the trained pipeline
pipe = sio.load("./model/loan_pipeline.skops", trusted=True)

def predict_loan(gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area):
    """
    Predicts home loan approval based on applicant features. Returns predicted loan approval decision.
    """
    try:
        if loan_amount_term <= 0:
            return "Error: Loan amount term must be greater than zero."

        total_income = applicant_income + coapplicant_income
        emi = loan_amount / loan_amount_term if loan_amount_term != 0 else 0

        features = pd.DataFrame([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area, total_income, emi
        ]], columns=[
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'TotalIncome', 'EMI'
        ])

        # Make prediction
        predicted_decision = pipe.predict(features)[0]
        label = f"Loan Approval Decision: {'Approved' if predicted_decision == 1 else 'Not Approved'}"
        return label
    except Exception as e:
        print(f"Error: {e}")
        return str(e)

inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Radio(["Yes", "No"], label="Married"),
    gr.Radio(["0", "1", "2", "3+"], label="Dependents"),
    gr.Radio(["Graduate", "Not Graduate"], label="Education"),
    gr.Radio(["Yes", "No"], label="Self Employed"),
    gr.Number(label="Applicant Income"),
    gr.Number(label="Coapplicant Income"),
    gr.Number(label="Loan Amount"),
    gr.Number(label="Loan Amount Term"),
    gr.Radio(["1", "0"], label="Credit History", info="1 for Good and 0 for Bad Credit History"),
    gr.Radio(["Urban", "Semiurban", "Rural"], label="Property Area")
]

outputs = gr.Label()

examples = [
    ["Male", "Yes", "1", "Graduate", "No", 6875, 0.0, 200, 360, "1", "Semiurban"],
    ["Female", "No", "0", "Graduate", "No", 2138, 0.0, 99, 360, "0", "Semiurban"],
    ["Male", "Yes", "2", "Graduate", "Yes", 8000, 3000, 250, 360, "1", "Semiurban"],
]

title = "Loan Approval Prediction"
description = "Enter the applicant details to predict loan approval."
article = "This app is part of the CICD-MLops project to train on automating ML model deployment using GitHub Actions."

gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()