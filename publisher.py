import paho.mqtt.client as mqtt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split

# Define the file path
file_path = 'train_data.csv'

# Read the file
train_df = pd.read_csv(file_path)

print(train_df.head())

# Assuming 'label' is the target column and the rest are features
X = train_df.drop(columns=['label'])  # Features
y = train_df['label']  # Target variable


# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Fit the model on the entire dataset
rf_model.fit(X, y)

# Make predictions on the same dataset (since we're not testing on separate data)
y_pred = rf_model.predict(X)

# Evaluate the model (on the same data)
print("Classification Report:")
print(classification_report(y, y_pred))

mcc = matthews_corrcoef(y, y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# Optionally, print feature importances
print("Feature Importances:")
print(rf_model.feature_importances_)



broker = "localhost"
port = 1883
topic = "dataflow"

client = mqtt.Client()
client.connect(broker, port)

#Publish the data
for index, row in train_df.iterrows():
    client.publish(topic, row.to_json())
    #print(f"{row.to_json()}")

result_topic = "resultsflow"

# Comentei porque vou enviar apenas o valor do MCC por enquanto
# # Publicar resultados no novo tópico
# for index, prediction in enumerate(y_pred):
#     result = {"index": index, "prediction": int(prediction)}
#     client.publish(result_topic, str(result))
#     print(f"Published to {result_topic}: {result}")

client.publish(result_topic, mcc)
print(f"Published MCC to {result_topic}: {mcc} ")

client.disconnect()



