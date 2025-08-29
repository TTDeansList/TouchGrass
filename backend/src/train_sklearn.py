import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from .preprocess import clean_text
from .rules import rule_flags
from .model_io import save_model

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    # weak label: violation if any rule trips
    flags = df["text"].apply(rule_flags)
    df["quality_violation_any"] = flags.apply(lambda f: int(any(f.values())))
    return df

def train_quality_baseline(df: pd.DataFrame):
    df = prepare_dataframe(df)
    X = df["text"]
    y = df["quality_violation_any"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=300))
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    print(classification_report(y_te, y_pred, digits=3))
    save_model(pipe)
    return pipe

if __name__ == "__main__":
    # Minimal demo training using a tiny inline dataset
    data = {
        "text": [
            "Buy my course at www.spam.com, best discount now!",
            "Food was fresh and staff were friendly. Would return.",
            "I haven't been here but I hate the owner.",
            "Try my promo code TODAY!",
            "Great coffee near the river, cozy ambience."
        ]
    }
    df = pd.DataFrame(data)
    train_quality_baseline(df)
