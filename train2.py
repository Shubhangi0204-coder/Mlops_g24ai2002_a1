import misc
from sklearn.kernel_ridge import KernelRidge

def main():
    # 1. Data Loading
    df = misc.load_data()
    
    # 2. Data Preprocessing and Splitting
    X_train, X_test, y_train, y_test = misc.preprocess_and_split(df)
    
    # 3. Model Definition (Using a common kernel like rbf)
    kr_model = KernelRidge(alpha=1.0, kernel='rbf', gamma=None)
    
    # 4. Training and Evaluation
    # Using the generic function from misc.py
    pipeline, mse = misc.train_and_evaluate_model(kr_model, X_train, X_test, y_train, y_test)
    
    # 5. Display Result
    print(f"Kernel Ridge Regressor - Average MSE on Test Set: {mse:.4f}")

if __name__ == "__main__":
    main()
