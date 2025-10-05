import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Data Loading Function
def load_data():
    """Loads the Boston Housing dataset following the specified instructions."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    # Read the data, separating by spaces, skipping the first 22 lines, no header
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    
    # Split into data and target as per instructions
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2] # MEDV is the target

    # Feature names
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    
    return df

# 2. Data Preprocessing and Splitting Function
def preprocess_and_split(df, target_col='MEDV', test_size=0.2, random_state=42):
    """Splits data into training/testing sets and returns features, target, and a scaler."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

# 3. Model Training and Evaluation Function (Generic)
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Trains a model within a pipeline (including a StandardScaler) and evaluates it.
    
    Args:
        model (estimator): The scikit-learn model to train (e.g., DecisionTreeRegressor).
    
    Returns:
        tuple: Trained pipeline and MSE on the test set.
    """
    # Create a pipeline with a standard scaler for preprocessing and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    return pipeline, mse
