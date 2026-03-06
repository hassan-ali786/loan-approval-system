from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/loan_model.pkl", "rb"))

@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
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
        result = "Approved ✅" if prediction==1 else "Rejected ❌"
        return render_template("result.html", result=result)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)