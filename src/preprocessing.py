import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    """
    Loads raw Titanic data, preprocesses it, and saves processed data.
    """

    df = pd.read_csv(input_path)

    # Drop unnecessary columns
    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    # Handle missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["Survived"] = y.values

    # Save processed dataset
    processed_df.to_csv(output_path, index=False)

    print("✅ Data preprocessing completed.")
