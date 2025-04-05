from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assume y_true is the true labels (0 for normal, 1 for attack) 
# and y_scores are the predicted scores from your model
# Example: y_true = [0, 1, 0, 1, 1, 0], y_scores = [0.2, 0.9, 0.1, 0.8, 0.7, 0.3]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Plot the Precision-Recall curve
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
