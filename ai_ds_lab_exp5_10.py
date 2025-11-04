# ===============================================
# Experiment 6: Naive Bayes Classifier
# ===============================================
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============================================
# Experiment 7: Decision Tree Classifier
# ===============================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pima = pd.read_csv("pima_indians_diabetes_sample.csv")
feature_cols = ['insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============================================
# Experiment 8: Hill Climbing Algorithm
# ===============================================
import math, random

def distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def total_distance(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

def swap_cities(tour):
    new_tour = tour[:]
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def hill_climbing(cities, max_iter):
    tour = list(range(len(cities)))
    random.shuffle(tour)
    current_distance = total_distance(tour, cities)
    for iteration in range(max_iter):
        new_tour = swap_cities(tour)
        new_distance = total_distance(new_tour, cities)
        if new_distance < current_distance:
            tour, current_distance = new_tour, new_distance
            print(f"Iteration {iteration + 1}: Distance = {current_distance}")
    return tour, current_distance

if __name__ == "__main__":
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    best_tour, best_distance = hill_climbing(cities, 10)
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)

# ===============================================
# Experiment 9: Correlation and Covariance
# ===============================================
df = pd.read_csv("iris.csv")
df.set_index("Id", inplace=True)
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
print("\nCovariance:\n", x.cov())
print("\nCorrelation:\n", x.corr(method='pearson'))

# ===============================================
# Experiment 10: PCA Feature Reduction
# ===============================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(scaled_features)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.title('2D PCA Plot - Breast Cancer Dataset')
plt.show()

df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
sns.heatmap(df_comp, cmap='plasma')
plt.title('PCA Components Heatmap')
plt.show()

pca = PCA(n_components=3)
x_pca = pca.fit_transform(scaled_features)
print("\nExplained Variance Ratio (3 components):", pca.explained_variance_ratio_)

