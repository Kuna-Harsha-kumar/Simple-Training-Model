# Simple Training Model

This repository contains two small Python machine-learning projects:

1. `trigram_language_model.py` trains a word-level trigram language model and generates text from prompts.
2. `sentiment_classification.py` trains and evaluates Naive Bayes and Logistic Regression classifiers for SST-2-style sentiment data.

## Requirements

Python 3.9 or newer is recommended. Install the dependencies with:

```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn openpyxl
```

## Trigram language model

Place the text corpora referenced by the script in the repository root:

- `leonardodavinci.txt`
- `edgarallanpoe.txt`

Run:

```bash
python trigram_language_model.py
```

The script trains one model for each text file and prints generated examples for several prompts. The model is implemented with Python's standard library and uses randomly selected next words weighted by trigram probability.

## Sentiment classification

Place the training and test datasets in the repository root:

- `train_sst2.xlsx`
- `test_sst2.xlsx`

Each dataset must contain `text` and `label` columns. CSV files are also supported if the paths in the script are changed from `.xlsx` to `.csv`.

Run:

```bash
python sentiment_classification.py
```

The script trains both classifiers, evaluates predictions when test labels are present, and writes:

- `nb_model.joblib`
- `lr_model.joblib`
- `nb_predictions.csv`
- `lr_predictions.csv`
- `Naive Bayes_confusion_matrix.png`
- `Logistic Regression_confusion_matrix.png`

Generated model files, predictions, and plots are runtime outputs and are not included in the repository.

## Notes

The repository does not include the referenced text corpora or SST-2 datasets. Add those files locally before running the corresponding scripts.
