Bike Rental Prediction

Predict daily bike rentals using historical data with an interactive Streamlit web app.
Problem Statement
Bike rental companies need to understand and forecast daily rental demand to optimize operations, inventory, and staffing. Predicting the number of bike rentals based on historical data, weather, and temporal features can help make data-driven business decisions.

 Objective
- Develop a machine learning model to predict daily bike rentals accurately.
- Provide an interactive Streamlit application where users can input features and obtain predictions in real-time.
- Explore the dataset to understand factors influencing bike rentals and visualize trends.

 Methodology / Steps Followed
1. Data Collection   
   - Used the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).  
   - Included a small sample dataset ('sample.csv`) in this repo for demonstration.

2. Data Preprocessing
   - Cleaned missing or inconsistent data.
   - Removed outliers  
   - Encoded categorical features and scaled numerical features.  
   - Split the dataset into training and testing sets.

3. Model Building & Evaluation 
   - Trained machine learning model.
   - Evaluated performance using metrics such as RMSE, MSE, and R².  

5. Deployment  
   - Built an interactive Streamlit app (`bike.py`) to input features and predict bike rentals.  
   - Added images and visual aids to improve user experience.

6. Documentation & Testing
   - Organized notebooks  with clear explanations.  

 Dataset
- Original dataset: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)  
- Sample dataset: `sample.csv` (included in repo)

