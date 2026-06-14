from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/loan_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            features = [
                int(request.form['Gender']),
                int(request.form['Married']),
                int(request.form['Dependents']),
                int(request.form['Education']),
                int(request.form['Self_Employed']),
                float(request.form['ApplicantIncome']),
                float(request.form['CoapplicantIncome']),
                float(request.form['LoanAmount']),
                float(request.form['Loan_Amount_Term']),
                int(request.form['Credit_History']),
                int(request.form['Property_Area'])
            ]

            prediction = model.predict([features])[0]
            proba_array = model.predict_proba([features])[0]

            # classes_ = [0, 1] → index 1 = approved probability
            approved_prob = round(float(proba_array[1]) * 100, 1)

            result = "Approved" if prediction == 1 else "Rejected"
            probability = approved_prob if prediction == 1 else round(float(proba_array[0]) * 100, 1)

            return render_template("result.html", result=result, probability=probability)

        except Exception as e:
            print("ERROR:", e)
            error_msg = "Invalid input. Please fill all fields correctly."
            return render_template("index.html", error=error_msg)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)