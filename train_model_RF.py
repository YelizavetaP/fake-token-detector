import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib


def train_and_save_scam_detector(
        data_csv_path: str,
        model_output_path: str,
        test_size: float = 0.2,
        random_state: int = 42
) -> None:
    """
    Train a scam token detection model and save the pipeline to disk.

    Args:
        data_csv_path (str): Path to the labeled CSV with 'name', 'url', and 'is_scam'.
        model_output_path (str): File path to save the trained model pipeline.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
    """
    # 1. Load the labeled dataset
    df = pd.read_csv(data_csv_path)
    if df.empty:
        raise ValueError("The dataset is empty. Please provide a valid CSV file.")

    # 2. Prepare the text feature by combining name and URL slug
    df['url_slug'] = df['url'].fillna('').apply(lambda x: x.split('/')[-1] if isinstance(x, str) else '') # виділити в URL останній сегмент після останнього тега
    df['text_features'] = (df['name'].fillna('') + ' ' + df['url_slug']).astype(str)

    # 3. Define features and target
    X = df['text_features']
    y = df['label']

    # 4. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    # 5. Create a pipeline with TF-IDF vectorizer and RandomForest classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ('clf', RandomForestClassifier(n_estimators=100, 
                                       random_state=random_state,
                                       n_jobs=-1))
    ])

    # 6. Train the model
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", 
          classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n",
          confusion_matrix(y_test, y_pred))

    # 7. Save the trained pipeline

    joblib.dump(pipeline, model_output_path)
    print(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    DATA_PATH = "data/final_labeled.csv"
    MODEL_PATH = "scam_token_detector.pkl"
    train_and_save_scam_detector(
        data_csv_path=DATA_PATH,
        model_output_path=MODEL_PATH,
        test_size=0.2,
        random_state=42
    )
