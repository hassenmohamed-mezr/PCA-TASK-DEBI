# PCA-Based Breast Cancer Classification

A machine learning comparison project that evaluates 11 classification
models on the **Breast Cancer Wisconsin Diagnostic dataset**, using
either the original 30 features or only 2 principal components generated
with Principal Component Analysis (PCA).

> **Disclaimer:** This is an educational machine learning project, not a
> medical diagnostic tool. Its predictions must not be used to make
> clinical decisions.

## Project Overview

The project investigates the trade-off between dimensionality reduction
and classification performance. It compares models trained on:

-   **30 original features** after standardization.
-   **2 PCA components** fitted using the standardized training data.

Both experiments use the same stratified train/test split to make the
comparison consistent.

## Dataset

The notebook loads the Breast Cancer Wisconsin Diagnostic dataset
directly from `scikit-learn`; no separate dataset download is required.

-   **Samples:** 569
-   **Input features:** 30 numeric features
-   **Classes:** 2 --- malignant and benign
-   **Training set:** 455 samples (80%)
-   **Test set:** 114 samples (20%)

Target mapping in the scikit-learn dataset:

-   `0` = malignant
-   `1` = benign

## Models Compared

The notebook evaluates these 11 classifiers:

1.  Logistic Regression
2.  Decision Tree
3.  K-Nearest Neighbors (KNN)
4.  Support Vector Machine (SVM)
5.  Gaussian Naive Bayes
6.  Voting Classifier (soft voting)
7.  Bagging Classifier
8.  Random Forest
9.  Gradient Boosting
10. AdaBoost
11. XGBoost

## Workflow

1.  Load the dataset into a Pandas DataFrame.
2.  Separate the feature matrix (`X`) from the target (`y`).
3.  Split the data into training and test sets using `test_size=0.2`,
    `random_state=42`, and stratification.
4.  Fit `StandardScaler` on the training set and use it to transform
    both sets.
5.  Train and evaluate all 11 classifiers using the 30 standardized
    features.
6.  Fit PCA with `n_components=2` on the standardized training data
    only.
7.  Transform the training and test sets into two principal components.
8.  Train and evaluate the same 11 classifiers using the two PCA
    components.
9.  Compare metrics and visualize confusion matrices, model scores, and
    the two-dimensional PCA projection.

## Evaluation Metrics

For each model and feature configuration, the notebook reports:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Classification report
-   Confusion matrix

**Metric note:** In the notebook's summary table, precision, recall, and
F1-score use `pos_label=1`; in this dataset, label `1` represents
**benign**. The classification report provides metrics for both classes.

## Results Snapshot

The results below are from the notebook's recorded run, using the
114-sample test set.

  Configuration          Best model by F1-score     Accuracy   F1-score
  ---------------------- ------------------------ ---------- ----------
  30 original features   Logistic Regression          98.25%     98.61%
  2 PCA components       SVM                          94.74%     95.89%

The two PCA components retain **63.36% of the total variance** in the
standardized training data. This reduces the feature count from 30 to 2,
but also discards some information. In this run, the mean accuracy
across the 11 models was **95.45%** with 30 features and **93.38%** with
2 PCA components.

These values are specific to this split and run; they are not a
guarantee of performance on other data.

## Visualizations

The notebook generates:

-   Confusion matrices for each model in both experiments.
-   A grouped bar chart comparing accuracy across the two feature
    configurations.
-   A grouped bar chart comparing F1-score across the two
    configurations.
-   A 2D PCA scatter plot showing the samples by class.
-   A results table and an accuracy comparison table.

## Requirements

-   Python 3.9 or later recommended
-   Jupyter Notebook or JupyterLab
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   scikit-learn
-   XGBoost

## Run the Project

1.  Clone the repository:

    ``` bash
    git clone https://github.com/hassenmohamed-mezr/PCA-TASK-DEBI.git
    cd PCA-TASK-DEBI
    ```

2.  (Recommended) Create and activate a virtual environment:

    ``` bash
    python -m venv .venv
    ```

    **Windows PowerShell:**

    ``` powershell
    .\.venv\Scripts\Activate.ps1
    ```

3.  Install the dependencies:

    ``` bash
    python -m pip install --upgrade pip
    python -m pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
    ```

4.  Launch Jupyter:

    ``` bash
    jupyter notebook
    ```

5.  Open `PCA_Task.ipynb` and run the cells from top to bottom.

## Repository Structure

``` text
PCA-TASK-DEBI/
├── PCA_Task.ipynb   # Data preparation, experiments, evaluation, and visualizations
├── LICENSE          # Project license
└── README.md        # Project documentation
```

## Reproducibility Notes

-   The train/test split uses `random_state=42` and stratification.
-   The scaler is fitted on the training data only.
-   PCA is fitted on the scaled training data only; the test data is
    transformed with the already-fitted PCA model.
-   Model scores can vary if the code, library versions, or model
    settings change.

## Future Improvements

-   Add cross-validation for more robust model comparison.
-   Tune model hyperparameters instead of relying only on the current
    settings.
-   Report per-class metrics prominently, especially malignant-class
    recall, for a more complete evaluation.
-   Save experiment outputs and library versions to make results easier
    to reproduce.

## Author

**Hassen Mohamed Hassen**

GitHub: [@hassenmohamed-mezr](https://github.com/hassenmohamed-mezr)
