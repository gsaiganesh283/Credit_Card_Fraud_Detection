# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using various classification algorithms and techniques to handle imbalanced datasets.

## Overview

This project implements a comprehensive fraud detection system that analyzes credit card transaction data to identify fraudulent activities. The dataset is highly imbalanced with only 2 fraudulent transactions compared to 991 legitimate ones, requiring specialized techniques for balanced model training.

## Dataset

- **File**: `creditcard.csv`
- **Features**: Multiple anonymized V features (V1-V27) plus Time and Amount
- **Class Distribution**: Imbalanced (mostly legitimate transactions with few frauds)

## Project Structure

```
Credit_Card_Fraud_Detection/
├── Credit_Card_Fraud_Detection.ipynb  # Main analysis and modeling notebook
├── creditcard.csv                      # Transaction dataset
└── README.md                           # This file
```

## Key Features

### Data Exploration & Preprocessing
- Loading and exploring credit card transaction data
- Handling missing values and duplicates
- Statistical analysis of features

### Feature Engineering
- Time feature extraction (hours, minutes)
- Amount logarithmic transformation for better visualization
- Feature selection based on correlation analysis
- Creating engineered features for improved model performance

### Handling Class Imbalance
- **RandomOverSampler**: Oversampling minority class
- **SMOTE**: Synthetic Minority Over-sampling Technique

### Models Implemented
- Random Forest Classifier
- Logistic Regression (with SMOTE pipeline)

## Requirements

The project uses the following libraries:
- pandas
- numpy
- scikit-learn
- imbalanced-learn (imblearn)
- seaborn
- matplotlib

Install the required packages:
```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
```

## Methodology

### 1. Data Loading & Exploration
- Load transaction data using pandas
- Check dataset shape, duplicates, and missing values
- Analyze class distribution

### 2. Feature Engineering & Splitting
- Separate features (X) from labels (y)
- Split data into 80% training and 20% testing sets with stratification
- Apply Random Over Sampler to balance training data

### 3. Model Training
- Train Random Forest Classifier on resampled data
- Configuration: 100 estimators, random_state=42

### 4. Model Evaluation
- Confusion Matrix analysis
- Classification Report (precision, recall, F1-score)
- Accuracy Score
- Additional metrics: MAE, MSE, RMSE, R² Score

## Model Performance

The notebook includes comprehensive evaluation metrics:
- **Confusion Matrix**: True/False positives and negatives
- **Precision & Recall**: Performance on fraud detection
- **Accuracy Score**: Overall model correctness
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**: Model fit quality

## Visualization & Analysis

The project includes extensive visualizations:
- Class distribution count plots
- Distribution of transactions by hours and minutes
- Amount distribution by fraud class (box plots)
- Feature correlation heatmaps
- Distribution plots for all V features

## Notebook Sections

1. **Importing Required Packages** - Load necessary libraries
2. **Data Exploration and Preprocessing** - Initial data analysis
3. **Feature Engineering and Splitting** - Prepare features for modeling
4. **Training the Random Forest Classifier** - Model training
5. **Model Evaluation** - Performance assessment
6. **Using SMOTE** - Advanced imbalanced data handling
7. **Time Features** - Extract temporal patterns
8. **Feature Selection** - Select most relevant features
9. **Performance Metrics** - Comprehensive evaluation

## Usage

Run the Jupyter notebook to execute the complete analysis:
```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

Or use VS Code:
```bash
code Credit_Card_Fraud_Detection.ipynb
```

## Key Insights

- Fraud transactions show different patterns in time and amount compared to legitimate transactions
- Certain V features are more discriminative for fraud detection
- Handling class imbalance is critical for effective fraud detection
- Both Random Over Sampling and SMOTE are effective techniques for this problem

## Future Improvements

- Implement additional models (XGBoost, Gradient Boosting)
- Cross-validation for more robust evaluation
- Hyperparameter tuning for optimal performance
- Feature selection using advanced techniques (feature importance, mutual information)
- Cost-sensitive learning to prioritize fraud detection
- Real-time prediction pipeline

## Author

gsaiganesh283

## License

This project is open-source and available for educational purposes.
