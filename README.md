# Heart Attack Risk Prediction (Simple Neural Network)

This project uses a **neural network built from scratch** to predict heart attack risk based on patient data.

##  About

We use a simple neural network with:

- One layer (perceptron)
- Sigmoid activation
- Manual gradient descent (no ML libraries)

## 📁 Dataset

The data comes from `heart.csv` and includes 13 medical features like age, blood pressure, cholesterol, and more. The target is:

- `output`: 1 = risk of heart disease, 0 = no risk

acess: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
obs - the last column named "target" on original database was exchanged for "output"

## 🚀 How to Run

1. Make sure `heart.csv` is in the same folder.
2. Install required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
