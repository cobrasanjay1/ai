# ===============================
# MATRIX OPERATIONS
# ===============================
import numpy as np

def get_matrix(rows, cols, name):
    print(f"Enter elements of {name} (space-separated row-wise):")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i + 1}: ").split()))
        matrix.append(row)
    return np.array(matrix)

# Input matrices
r1 = int(input("Enter number of rows for matrix 1: "))
c1 = int(input("Enter number of columns for matrix 1: "))
matrix1 = get_matrix(r1, c1, "matrix 1")

r2 = int(input("Enter number of rows for matrix 2: "))
c2 = int(input("Enter number of columns for matrix 2: "))
matrix2 = get_matrix(r2, c2, "matrix 2")

# Addition
if matrix1.shape == matrix2.shape:
    print("\nAddition:\n", matrix1 + matrix2)
else:
    print("Addition not possible")

# Subtraction
if matrix1.shape == matrix2.shape:
    print("\nSubtraction:\n", matrix1 - matrix2)
else:
    print("Subtraction not possible")

# Multiplication
if c1 == r2:
    print("\nMultiplication:\n", np.dot(matrix1, matrix2))
else:
    print("\nMultiplication not possible")

# Element-wise Division
if matrix1.shape == matrix2.shape:
    try:
        print("\nElement-wise Division:\n", matrix1 / matrix2)
    except ZeroDivisionError:
        print("Division by zero detected")
else:
    print("\nDivision not possible")

# Inverse
def matrix_inverse(matrix, name):
    if matrix.shape[0] != matrix.shape[1]:
        print(f"\n{name} is not square, Inverse not possible.")
        return
    try:
        inv = np.linalg.inv(matrix)
        print(f"\nInverse of {name}:\n", inv)
    except np.linalg.LinAlgError:
        print(f"\n{name} is not invertible.")

matrix_inverse(matrix1, "Matrix 1")
matrix_inverse(matrix2, "Matrix 2")

# ===============================
# PLOTLY BAR CHART
# ===============================
import plotly.express as px
import pandas as pd

data = {'product': ['A', 'B', 'C', 'D'], 'Sales': [120, 340, 290, 410]}
df = pd.DataFrame(data)
fig = px.bar(df, x='product', y='Sales', color='product', title='Product Sales')
fig.show()

# ===============================
# PANDAS DATAFRAME OPERATIONS
# ===============================
data = {'Name': ['Amal', 'Sree', 'Sanooj', 'Jinsil'],
        'Age': [25, 30, 35, 40],
        'Salary': [50000, 60000, 70000, 80000]}
df = pd.DataFrame(data)
print("Data Summary:")
print(df.describe())

print("\nEmployees with Salary > 60000:")
print(df[df['Salary'] > 60000])

df['tax'] = df['Salary'] * 0.2
df['yr_slr'] = df['Salary'] * 12
df['anu_income'] = df['yr_slr'] - df['tax']
print(df)

# ===============================
# SVM - HEART DISEASE PREDICTION
# ===============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("heart-ds.csv")
data['target'] = np.where(data['target'] > 0, 1, 0)

X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

