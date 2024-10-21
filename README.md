<h1>Customer Churn Prediction with H2O AutoML</h1>
This project aims to predict customer churn using an automated machine learning framework, H2O AutoML. Churn refers to the rate at which customers discontinue their relationship with a company, product, or service over a specific period. Understanding and predicting churn is essential for businesses, particularly in subscription-based models, to retain customers and minimize revenue loss.
<br>
<h3>What is Churn?</h3><br>
Churn is a vital business metric that tracks the rate at which customers stop using services. In this project, we address two types of churn:
<ol>
<li>Customer Churn: When a customer cancels their subscription or stops using a service.</li>
<li>Revenue Churn: When revenue decreases due to cancellations, downgrades, or non-renewals. This is particularly critical for subscription-driven companies.</li></ol>
<h3>Project Overview</h3>
In this project, I utilized H2O AutoML to automatically build, train, and evaluate various machine learning models to predict customer churn. The aim is to develop a reliable model to help businesses identify customers who are likely to churn, allowing them to take preventative actions.

<h3>Steps in the Project:</h3>
<h4>Setting Up the Environment:</h4>

Installed Java and necessary H2O packages as the H2O platform requires Java for model execution.
bash
Copy code
# Install H2O (if not already installed)
pip install h2o<br>
Loading the Dataset:

I used a dataset containing various customer attributes (e.g., gender, senior citizen status, tenure, payment method) and the churn status (yes/no).
python
Copy code
import h2o
from h2o.estimators import H2OAutoML

# Start H2O
h2o.init()

# Load dataset
churn_data = h2o.import_file("path/to/churn_data.csv")
Splitting the Data:

The dataset was split into training, validation, and test sets to ensure effective model training, validation during training, and evaluation.
python
Copy code
# Split the data
train, valid, test = churn_data.split_frame(ratios=[.7, .15], seed=1234)<br>
Training with H2O AutoML:

I leveraged H2O AutoML to automatically train multiple models, including GBM, GLM, and XGBoost.
python
Copy code
# Specify the response variable and features
y = "Churn"
x = churn_data.columns
x.remove(y)

# Train the AutoML model
aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
Model Selection and Predictions:

The best-performing model was an XGBoost model, which demonstrated the highest accuracy and overall performance.
python
Copy code
# Get the best model
best_model = aml.leader

# Make predictions on the test set
churn_pred = best_model.predict(test)
<h3>Model Performance:</h3>

The model's performance on the test set was evaluated using the aml.leader.model_performance(test) function.
python
Copy code
# Evaluate model performance
performance = best_model.model_performance(test)
print(performance)
<h3>Key Outcomes</h3>
The automated approach using H2O AutoML provided valuable insights into customer churn patterns, offering clear indicators of why customers might leave. With these insights, businesses can take data-driven actions to mitigate churn risks and improve customer retention.

<h3>Why This Matters</h3>
Churn prediction is not just about identifying which customers will leave but also understanding the reasons behind it. By analyzing customer behavior patterns, businesses can enhance the overall customer experience, leading to increased customer satisfaction and loyalty.

<h3>Conclusion</h3>
This project demonstrates the power of H2O AutoML in simplifying the model-building process for predicting customer churn. By automating the model selection process, we can efficiently address churn and help businesses make data-driven decisions to retain customers.

<h2>Tools and Technologies Used</h2>
<ol>
  <li>H2O AutoML</li>
<li>XGBoost</li>
<li>Python</li>
<li>Jupiter Nootebook for interactive analysis and development</li>
<li>Sklearn for data preprocessing</li>

<h3>How to Run the Project</h3>
Clone the repository to your local machine.

bash
Copy code
git clone https://github.com/JIGEESHA-ANAGANI/AUTOML/blob/main/AUTOML%20(1).ipynb
Install the necessary dependencies

bash
Copy code
pip install -r requirements.txt
Open and run the churn_prediction.ipynb notebook to follow the entire workflow, from loading the dataset to training and evaluating the model.
![predictions](https://github.com/JIGEESHA-ANAGANI/AUTOML/blob/main/XGBOOST.png)
