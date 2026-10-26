# Predicting Presence and Absence of Kyphosis

Individual AI & Data Science II mini-project, recreated to match the **submitted Experiment 10 report**.

## Problem
Classify whether kyphosis was `present` or `absent` after corrective spinal surgery using three numeric inputs: `Age` (months), `Number` (vertebrae involved), `Start` (topmost vertebra operated on).

## Dataset
81 observations, 4 columns, no missing values; 64 absent and 17 present. Dataset is a classic teaching example from the R `rpart` package. Data source: https://github.com/vincentarelbundock/Rdatasets/blob/master/csv/rpart/kyphosis.csv . The `rownames` export column was omitted.

## Models
- Decision Tree classifier
- Random Forest classifier (100 trees)

Split: 70% training, 30% test, stratified, random_state=42. Classification metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrices. The small dataset means results are unstable; the app is for **educational demonstration only, not medical diagnosis**.

## Reproducing
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open `notebooks/kyphosis_analysis.ipynb` and run all cells to regenerate EDA, metrics and models. Pretrained model files are included for immediate app use.

## File layout
- `data/kyphosis.csv`: dataset
- `notebooks/kyphosis_analysis.ipynb`: EDA, training and evaluation
- `models/`: trained model files
- `outputs/metrics.json`: saved metrics from this build
- `app.py`: interactive Streamlit demo
- `START_HERE.txt`: setup instructions

## Important distinction from submitted report
The **original submitted report** recorded 84% accuracy for both models, with confusion matrix `[[18,1],[3,3]]`. The report did not fully specify the random seed and hyperparameters. This reproducible implementation uses explicit settings, so its test metrics can differ. Do not replace or mislabel the original report's results without explaining the change.

## Future work
Cross-validation, class-imbalance handling, and tuning for improved `present` recall; independent external validation would be required before any medical use.
