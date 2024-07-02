# Big Data Flight Delay Prediction with Spark

## Overview
This project analyzes flight data using big data techniques. It includes data cleaning, processing, and machine learning models to predict flight arrival delays.

## Team Members
- Dániel Laki
- Yacine Merioua
- Federico Paschetta

## Project Structure
```
Big Data Project/
│
├── data/
│   ├── 1987.csv
│   ├── 2008.csv
│   └── airports.csv
│
├── src/
│   └── main/
│       └── python/
│           ├── project_num.py
│           └── project.py
│
├── authors.txt
├── requirements.txt
└── report.pdf
```

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning models:
  - Linear Regression
  - Random Forest Regression (commented out due to performance issues)
- Two versions of the model:
  - "Numerical" version (project_num.py): Uses airport coordinates
  - "Categorical" version (project.py): Uses origin and destination as categorical variables

## Requirements
- Python 3.x
- PySpark 3.5.0
- Matplotlib 3.8.2
- Seaborn 0.13.1
- Pandas 2.1.4

## Installation
1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the script, use the following command in the terminal:

For the numerical model:
```
spark-submit --master local[*] src/main/python/project_num.py 1987.csv
```

For the categorical model:
```
spark-submit --master local[*] src/main/python/project.py 1987.csv
```

You can also run the script with multiple datasets:
```
spark-submit --master local[*] src/main/python/project_num.py 1987.csv 2008.csv
```

## Data Cleaning and Processing
- Removes forbidden variables
- Handles null values
- Removes duplicates
- Groups variables into categorical and numerical
- Performs One Hot Encoding on categorical variables
- Normalizes time variables

## Model Evaluation
The model uses the following metrics for evaluation:
- RMSE (Root Mean Square Error)
- R-squared
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- Explained Variance

## Results
- Categorical Model (1987 data):
  - RMSE: 15.27
  - R-squared: 65.03%
- Numerical Model (1987 data):
  - RMSE: 14.62
  - R-squared: 67.68%
- Categorical Model (2008 data):
  - RMSE: 10.05
  - R-squared: 93.23%

Note: The numerical model faced performance issues with the 2008 dataset.

## Known Issues
- The Random Forest Regression model may face memory issues on machines with limited resources.
- Cross-validation is implemented but commented out due to long execution times.

## Future Work
- Optimize the Random Forest Regression model for better performance
- Implement and optimize cross-validation
- Explore additional features or data sources to improve prediction accuracy


