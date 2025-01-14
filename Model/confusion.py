import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix

#Load model
logreg_model = joblib.load('../Model/logreg_model.pkl')

#Load test data
test_df = pd.read_csv('../Data_files/mcc_test.csv')

#Separate features and label
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

#Make predictions
y_pred = logreg_model.predict(X_test)

#Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix: SGD")
print(conf_matrix)

#Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix: SGD')
plt.show()