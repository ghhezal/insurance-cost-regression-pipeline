# Insurance Cost Regression Pipeline

A linear regression project that predicts medical insurance charges. The main goal was to implement the normal equation from scratch instead of just calling sklearn.

## What this project does

Takes a dataset of 1338 patients (age, BMI, smoker status, sex, region, number of children) and predicts their insurance charges.

## Project structure

```
insurance-cost-regression-pipeline/
├── data/
│   └── insurance.csv
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data_preprocessing.py
│   └── evaluation.py
├── notebook.ipynb
├── main.py
└── README.md
```

## What I found in the data

- Charges is right-skewed, so I log-transformed the target before modeling
- Smoker is the most important feature by far, you can see it clearly in the pairplot as three separate diagonal bands
- Sex and region barely affect cost, so I dropped them
- Age and BMI are the strongest numeric features

I ended up using only three features: age, bmi, and is_smoker.

## The model

I implemented the normal equation from scratch:

```
θ = (XᵀX)⁻¹ Xᵀy
```

To handle the intercept inside the matrix operation, I prepend a column of ones to X:

```python
Xb = np.c_[np.ones((X.shape[0], 1)), X]
A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
```

## Results

Trained on 75% of the data, tested on the remaining 25%.

| Model | Train R² | Test R² |
|---|---|---|
| sklearn LinearRegression | 0.7662 | 0.7481 |
| Custom Normal Equation | 0.7662 | 0.7481 |

Scores match exactly, which confirms the implementation is correct.

## What the plots showed

The predicted vs actual plot shows two bands instead of one clean diagonal. The model captures the smoker effect but not fully.

The residual plot is not random, there is a clear pattern. The likely cause is that among smokers, cost scales differently with BMI, and a simple linear model cannot capture that without an explicit interaction feature.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

## Stack

Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn (for train/test split and baseline comparison only)
