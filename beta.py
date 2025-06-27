import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic customer dataset
def create_synthetic_data(n_samples=1000):
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 60, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One-year', 'Two-year'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber', 'None'], n_samples),
        'total_charges': np.random.uniform(50, 5000, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% churn rate
    }
    return pd.DataFrame(data)

# 2. Data preprocessing
def preprocess_data(df):
    # Handle missing values (if any)
    df = df.dropna()
    
    # Encode categorical variables
    categorical_cols = ['contract_type', 'internet_service']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Define features and target
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test, y_train, y_test, scaler

# 3. Train and evaluate models
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    
    print(f"\n{model_name} Performance:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return model, precision

# 4. Plot feature importance (for Decision Tree)
def plot_feature_importance(model, features, model_name):
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp)
    plt.title(f'Feature Importance - {model_name}')
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Create dataset
    df = create_synthetic_data()
    print("Dataset Shape:", df.shape)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Initialize models
    log_reg = LogisticRegression(random_state=42, C=1.0)
    decision_tree = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    # Train and evaluate Logistic Regression
    log_reg_model, log_reg_precision = train_evaluate_model(
        log_reg, X_train, X_test, y_train, y_test, "Logistic Regression"
    )
    
    # Train and evaluate Decision Tree
    dt_model, dt_precision = train_evaluate_model(
        decision_tree, X_train, X_test, y_train, y_test, "Decision Tree"
    )
    
    # Plot feature importance for Decision Tree
    plot_feature_importance(decision_tree, X_train.columns, "Decision Tree")
    
    # Save the best model (based on precision)
    best_model = log_reg_model if log_reg_precision > dt_precision else dt_model
    joblib.dump(best_model, 'best_churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nBest model and scaler saved as 'best_churn_model.pkl' and 'scaler.pkl'")
