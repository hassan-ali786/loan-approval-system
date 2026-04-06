# House Price Prediction Web Application

An end-to-end Machine Learning web application that predicts house prices based on user inputs such as area, number of bedrooms, and bathrooms.  
The system provides real-time predictions through a professional web interface.

---

## Project Overview

This application allows users to enter property details and predicts house prices using a trained machine learning model.  
It follows a complete ML lifecycle:

- Data preprocessing and feature handling  
- Model training using regression algorithms  
- Model evaluation and saving  
- Deployment using Flask  
- Professional frontend integration for real-time predictions  

---

## Features

- **Machine Learning Prediction:** Predicts house prices using a trained regression model  
- **Interactive Web Interface:** Simple and responsive form for entering property details  
- **Instant Prediction:** Users receive predicted house prices immediately after submitting the form  
- **Clean Project Architecture:** Well-organized folders for data, models, frontend, and backend code  
- **Flask Web Integration:** Backend built using the Flask web framework  
- **Deployment Ready:** Can be easily deployed to cloud platforms  

---

## Dataset Information

**Dataset:** Housing dataset used for training the regression model  

**Features:**  

- Area (square feet)  
- Number of Bedrooms  
- Number of Bathrooms  

**Target:** House Price  

The model predicts house prices based on these features.

---

## Project Structure

```bash
house-price-prediction/
├── data/
│   └── house_data.csv
├── model/
│   └── house_price_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── css/style.css
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation Guide

1. Clone the repository:

```bash
git clone https://github.com/hassan-ali786/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the machine learning model:

```bash
python train_model.py
```

This will create the trained model file:

```
model/house_price_model.pkl
```

4. Run the web application:

```bash
python app.py
```

5. Open a browser at `http://127.0.0.1:5000`, enter house details, and click **Predict**.

---

## Key Features

- Real-time house price prediction  
- Professional and user-friendly interface  
- Modular and production-ready architecture  
- Easy deployment on cloud platforms  

---

## Technology Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)  
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)  
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)  
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)  
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)  
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)  

---

## Requirements

```
Flask==2.3.5
pandas==2.1.1
scikit-learn==1.3.2
numpy==1.26.5
joblib==1.3.2
```

---

## Future Improvements

- Add more housing features for better prediction accuracy  
- Implement advanced regression models  
- Deploy the application on cloud platforms  
- Store prediction history in a database  
- Add visualization dashboard for housing insights  

---

## Author

Hassan Ali  
 Data Scientist and Machine Learning Engineer  

GitHub: https://github.com/hassan-ali786  

---

## Application Screenshot

![House Price Web App](assets/homepage.png) 
