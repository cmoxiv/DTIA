
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train a decision tree
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Plot and save as SVG
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.savefig("iris_tree.pdf", format='pdf')  # 🔥 Save as SVG
plt.close()

print("Saved tree to iris_tree.svg ✅")
