# ECG Heartbeat Classification with ANN

 classify pre-segmented ECG heartbeat samples into five categories using feedforward Artificial Neural Networks.

> Educational project only. This model is not a medical diagnostic tool.

## Dataset setup

Download the Kaggle ECG Heartbeat Categorization dataset:
https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Place these files in the `dataset/` directory:
- `mitbih_train.csv`
- `mitbih_test.csv`

Each sample has 187 signal features and a final label in the notebook dataset files.

## Project layout

```text
ECG_Heartbeat_Classification_Clean/
├── dataset/
│   ├── mitbih_train.csv
│   └── mitbih_test.csv
├── notebooks/
│   └── ECG_Heartbeat_Classification_Clean.ipynb
├── models/                 # Saved ANN and scaler
├── results/
│   ├── plots/
│   └── reports/
└── app/
    ├── app.py
    ├── templates/
    │   └── index.html
    └── static/
```

## Setup

Open a terminal in the project root:

```bash
pip install -r requirements.txt
```

Open `notebooks/ECG_Heartbeat_Classification_Clean.ipynb` in Jupyter Notebook or JupyterLab and run cells from top to bottom.

The notebook:
1. Loads and explores the data.
2. Splits training data into train/validation subsets.
3. Scales features using a scaler fitted only on training data.
4. Trains baseline, deep, Swish, and SMOTE ANN experiments.
5. Selects the model using validation Macro F1.
6. Evaluates the selected model once on the held-out test set.
7. Saves model, scaler, plots, and reports.

## Run the web application

First run the notebook through the save-model section to generate:

- `models/ecg_best_ann.keras`
- `models/ecg_scaler.pkl`
- `results/reports/sample_ecg.csv`

Then from the project root:

```bash
python app/app.py
```

Open the address printed in the terminal (normally `http://127.0.0.1:5000`) and upload the sample CSV. Input CSVs should have exactly 187 numeric columns and must not include the target label.

## Evaluation

Model selection uses validation Macro F1, which is useful when class counts are imbalanced. Final test accuracy, macro/weighted F1, per-class precision/recall, and confusion matrix are reported only after selection.

## Limitations

- ECG samples are pre-segmented; this notebook does not segment raw continuous recordings.
- A beat-level split may not represent patient-independent generalization.
- Class weighting and SMOTE are experimental approaches; SMOTE creates synthetic feature vectors that may not represent physiologically valid waveforms.
- Performance metrics do not establish clinical safety or diagnostic reliability.

## Flask Web Application

The Flask interface is available in `app/`. It uses a light pastel dashboard theme and displays the prediction, class probabilities, uploaded ECG waveform, and predicted class counts. The footer credits the project developer.

### Prepare a small CSV for the web demo

The Kaggle MIT-BIH CSV files contain 188 columns: 187 signal values followed by one class label. The web app accepts either 187 signal columns or 188 columns; when 188 columns are detected, the final label column is ignored and is not used as a model input.

For a quick demo, create a small sample file instead of uploading the entire training dataset:

```python
import pandas as pd

train_data = pd.read_csv("dataset/mitbih_train.csv", header=None)
train_data.head(100).to_csv("dataset/ecg_demo_sample.csv", index=False, header=False)
```

Upload `ecg_demo_sample.csv` in the web application. The app requires the trained files `models/ecg_best_ann.keras` and `models/ecg_scaler.pkl`, created by running the notebook.

Run from the project root:

```bash
python app/app.py
```

Then open `http://127.0.0.1:5000` in your browser.

**Note:** This application is an educational machine-learning demonstration and must not be used for medical diagnosis.
