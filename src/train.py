import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data
from model import train_model
from evaluation import evaluate_model


RAW_DATA_PATH = "../data/raw/train.csv"
PROCESSED_DATA_PATH = "../data/processed/processed_data.csv"


def main():
    # Step 1: Preprocess data
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)

    # Step 2: Load processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    evaluate_model(model, X_test, y_test, X_train, y_train)


if __name__ == "__main__":
    main()
