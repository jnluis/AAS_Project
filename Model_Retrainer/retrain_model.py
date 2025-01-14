import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import paho.mqtt.client as mqtt
import json

# test_df = pd.read_csv('../Data_files/test_data.csv')


# # Load the saved model
# logreg_model = joblib.load('../Model/logreg_model.pkl')

# # Separate features (X_test) and label (y_test) from the test data
# X_test = test_df.drop(columns=['label'])
# y_test = test_df['label']
# y_pred = logreg_model.predict(X_test)

# # Evaluate the model on the test data
# print("Model Performance on Test Data (BEFORE):")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))


# # New data point for retraining
# new_data_df = pd.read_csv('retrain.csv')

# # Separate features (X) and label (y) from the new data
# X_new = new_data_df.drop(columns=['label'])
# y_new = new_data_df['label']

# # Use partial_fit to update the model with the new data
# logreg_model.partial_fit(X_new, y_new)

# # Save the updated model
# joblib.dump(logreg_model, '../Model/logreg_model.pkl')
# print("Model updated and saved.")

# test_df = pd.read_csv('../Data_files/test_data.csv')

# # Separate features (X_test) and label (y_test) from the test data
# X_test = test_df.drop(columns=['label'])
# y_test = test_df['label']

# # Make predictions on the test set
# y_pred = logreg_model.predict(X_test)

# # Evaluate the model on the test data
# print("Model Performance on Test Data (AFTER):")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(conf_matrix)

# # Plot Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()

# # Specify the path to the text file
# accuracy_file_path = '../Model/final_accuracy.txt'
# # Write the final accuracy to the text file
# with open(accuracy_file_path, 'a') as f:
#     f.write(f"Final Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

# print(f"Final accuracy saved to {accuracy_file_path}")



import sqlite3
import json
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

# Database setup
db_file = "mqtt_data.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()


def retrain_model(counter):
    # Load the saved model
    logreg_model = joblib.load('../Model/logreg_model.pkl')
    
    # Export data to CSV
    export_data_to_csv()

    # Load the new data
    new_data_df = pd.read_csv('retrain.csv')

    # Separate features (X) and label (y) from the new data
    X_new = new_data_df.drop(columns=['label'])
    y_new = new_data_df['label']

    # Retrain the model with the new data
    logreg_model.partial_fit(X_new, y_new)

    # Save the updated model
    joblib.dump(logreg_model, '../Model/logreg_model.pkl')
    print("Model updated and saved.")

    # Evaluate the model on the new data
    test_df = pd.read_csv('../Data_files/mcc_test.csv')
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    y_pred = logreg_model.predict(X_test)

    print("Model Performance on Test Data (AFTER):")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #Write the final accuracy to the text file
    accuracy_file_path = '../Model/final_accuracy.txt'
    with open(accuracy_file_path, 'a') as f:
        #Write the final accuracy to the text file with counter
        f.write(f"Final Accuracy: {accuracy_score(y_test, y_pred):.5f} - {counter}\n")

    print("Cleaning up the database...")
    delete_data_from_db()

    
def export_data_to_csv(db_file="mqtt_data.db", csv_file="retrain.csv"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Read all rows from the dataflow table
    cursor.execute("SELECT * FROM dataflow")
    rows = cursor.fetchall()

    # Convert to a DataFrame
    df = pd.DataFrame(rows, columns=["id", "data"])

    # Extract the JSON data and convert it to columns
    df_data = pd.json_normalize(df['data'].apply(eval))

    # Save to CSV
    df_data.to_csv(csv_file, index=False)
    print(f"Exported data to {csv_file}")

    # Close the connection
    conn.close()

def delete_data_from_db(db_file="mqtt_data.db"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Delete all rows from the dataflow table
    cursor.execute("DELETE FROM dataflow")
    conn.commit()
    print("Deleted all data from the database.")

    # Close the connection
    conn.close()


# Global counter variable
counter = 0

# Callback when a message is received
def on_message(client, userdata, msg):
    global counter
    print(f"Received message from {msg.topic}: {msg.payload.decode()}")

    # Parse the received message
    data = json.loads(msg.payload.decode())
    if data.get("retrain") is True:
        print("Retraining the model...")
        retrain_model(counter)
        print("Model retrained successfully.")
        counter += 1

# Broker details
broker = "localhost"
port = 12000
topic = "retrain"
# Create an MQTT client instance
client = mqtt.Client(protocol=mqtt.MQTTv311)

# Assign the on_message callback
client.on_message = on_message

# Connect to the broker
client.connect(broker, port)

# Subscribe to the topic
client.subscribe(topic)

# Start the loop to process received messages
print(f"Subscribed to {topic} and waiting for messages...")
client.loop_forever()
