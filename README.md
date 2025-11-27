 SOFE3370 Final Project 
 Battery Pack SOH Prediction with Linear Regression
 Group 4


 Project Overview
This project develops a Linear Regression model to predict the State of Health (SOH) of battery packs using voltage data (U1–U21) from the PulseBat Dataset.  
The system preprocesses the dataset, aggregates voltage readings, trains a regression model, and evaluates its predictive performance.

The project consists of two main components:  
- **Dataset Preprocessing and Aggregation:** Automatically detects voltage columns U1–U21, removes missing or invalid values, calculates the Pack SOH as the normalized average of all cells, and exports a cleaned dataset ready for training.  
- **Model Training and Evaluation:** Splits data into training (80%) and testing (20%) subsets, applies feature scaling using StandardScaler, trains a LinearRegression model from scikit-learn, evaluates model accuracy (R², MSE, MAE), and generates a plot comparing Actual vs Predicted SOH.


Program Execution and Results
The system requires Python version 3.9 or later for execution.  
To install the required libraries, run the following command in the terminal:

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl flask google-generativeai
```

Both **train_linear_regression.py** and **PulseBat Dataset (1).xlsx** must be located in the same directory.  

Execute the program in VS Code or the terminal with the command:

```bash
python train_linear_regression.py
```

After running, the program automatically creates a **results** folder containing three essential files:  
- **preprocessed_dataset.csv** → cleaned dataset ready for training  
- **model_metrics.txt** → accuracy results (R², MSE, MAE)  
- **soh_prediction_plot.png** → comparison between actual and predicted SOH values  

**Example output:**  
R²: 0.5981  
MSE: 0.002076  
MAE: 0.036552  

These results demonstrate that the model effectively predicts the State of Health with acceptable accuracy.

To run the chatbot as a CLI application:
```bash
py bot.py
```
To run the chatbot with a web interface
```bash
py server.py
```

 Project Directory Structure
FINAL_PROJECT_ALG/  
 
    train_linear_regression.py  
    PulseBat Dataset (1).xlsx   
    results/  
          preprocessed_dataset.csv  
          model_metrics.txt  
          soh_prediction_plot.png  
     README.md - Project documentation  
     explanatory_document.pdf 










