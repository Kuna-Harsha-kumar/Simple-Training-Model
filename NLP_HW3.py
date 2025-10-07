import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Conversion of excel file to csv file to make the dataset easily readable
def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Basic text preprocessing
def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)  
    return text

# Neutral words
NEGATION = {"not", "no", "never", "n't"}
# Positive words
POS_LEX = set(["good", "great", "excellent", "amazing", "love", "liked", "awesome", "best"])
# Negative words
NEG_LEX = set(["bad", "terrible", "worst", "hate", "awful", "boring", "disappoint"])

def extract_features(text):
    text = preprocess_text(text)
    tokens = re.findall(r"\w+", text)
    num_tokens = len(tokens)
    avg_len = np.mean([len(t) for t in tokens]) if tokens else 0
    exclaims = text.count('!')
    questions = text.count('?')
    uppercase_words = sum(1 for t in re.findall(r"\b[A-Z]{2,}\b", text))
    negations = sum(1 for w in tokens if w in NEGATION)
    pos_count = sum(1 for w in tokens if w in POS_LEX)
    neg_count = sum(1 for w in tokens if w in NEG_LEX)
    return [num_tokens, avg_len, exclaims, questions, uppercase_words,
            negations, pos_count, neg_count]

FEATURE_NAMES = ['num_tokens','avg_len','exclaims','questions','uppercase_words','negations','pos_count','neg_count']

# Training Naive-Bayes
def train_NB_model(path_to_train_file):
    df = load_data(path_to_train_file)
    texts = [preprocess_text(t) for t in df['text'].astype(str).values]
    labels = df['label'].astype(int).values

    vect = CountVectorizer(ngram_range=(1,2), min_df=2)
    X = vect.fit_transform(texts)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)

    NB_model = {'vectorizer': vect, 'clf': clf}
    joblib.dump(NB_model, 'nb_model.joblib')
    return NB_model

# Testing Naive-Bayes
def test_NB_model(path_to_test_file, NB_model, out_path='nb_predictions.csv'):
    df = load_data(path_to_test_file)
    texts = [preprocess_text(t) for t in df['text'].astype(str).values]

    X = NB_model['vectorizer'].transform(texts)
    probs = NB_model['clf'].predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)

    df['prob_positive'] = probs
    df['prediction'] = preds
    df.to_csv(out_path, index=False)
    return df

# Training Logistic Regression
def train_LR_model(path_to_train_file):
    df = load_data(path_to_train_file)
    features = [extract_features(t) for t in df['text'].astype(str).values]
    X = pd.DataFrame(features, columns=FEATURE_NAMES)
    y = df['label'].astype(int)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    clf.fit(Xs, y)

    LR_model = {'scaler': scaler, 'clf': clf, 'feature_names': FEATURE_NAMES}
    joblib.dump(LR_model, 'lr_model.joblib')
    return LR_model

# Testing Logistic Regression
def test_LR_model(path_to_test_file, LR_model, out_path='lr_predictions.csv'):
    df = load_data(path_to_test_file)
    features = [extract_features(t) for t in df['text'].astype(str).values]
    X = pd.DataFrame(features, columns=LR_model['feature_names'])
    Xs = LR_model['scaler'].transform(X)

    probs = LR_model['clf'].predict_proba(Xs)[:,1]
    preds = (probs >= 0.5).astype(int)

    df['prob_positive'] = probs
    df['prediction'] = preds
    df.to_csv(out_path, index=False)
    return df

# Evaluating model
def evaluate_model(df_with_preds, model_name):
    if 'label' not in df_with_preds.columns:
        print(f"[{model_name}] No true labels in test file. Skipping evaluation.")
        return
    y_true = df_with_preds['label']
    y_pred = df_with_preds['prediction']
    probs = df_with_preds['prob_positive']

    print(f"\n {model_name} Evaluation  :  ")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, probs))

    # Confusion matrix labelling
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{model_name}_confusion_matrix.png")  # Save instead of showing
    plt.close()

# Main method
if __name__ == "__main__":
    train_file = "train_sst2.xlsx"
    test_file = "test_sst2.xlsx"

    # Naive Bayes
    NB_model = train_NB_model(train_file)
    df_nb = test_NB_model(test_file, NB_model, out_path="nb_predictions.csv")
    evaluate_model(df_nb, "Naive Bayes")

    # Logistic Regression
    LR_model = train_LR_model(train_file)
    df_lr = test_LR_model(test_file, LR_model, out_path="lr_predictions.csv")
    evaluate_model(df_lr, "Logistic Regression")
