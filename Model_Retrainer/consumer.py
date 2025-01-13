import paho.mqtt.client as mqtt
import sqlite3
import json
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

# Database setup
db_file = "mqtt_data.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create a table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS dataflow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL
)''')
conn.commit()

def calculate_mcc():
    #Load data from dataset from csv file
    df_data = pd.read_csv('../Data_files/mcc_test.csv')
    X = df_data.drop(columns=['label'])  # Features
    y = df_data['label']  # Target variable
    #Load model from file
    rf_model = joblib.load('../Model/logreg_model.pkl')
    # Make predictions
    y_pred = rf_model.predict(X)
    # Evaluate the model
    mcc = matthews_corrcoef(y, y_pred)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    return mcc

def calculate_accuracy():
    df_data = pd.read_csv('../Data_files/mcc_test.csv')
    X = df_data.drop(columns=['label'])  # Features
    y = df_data['label']  # Target variable
    #Load model from file
    rf_model = joblib.load('../Model/logreg_model.pkl')
    # Make predictions
    y_pred = rf_model.predict(X)
    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy




# Broker details
broker = "localhost"
port = 12000
topic = "dataflow"
new_topic = "grafana"

messages_count = 0

# Callback when a message is received
def on_message(client, userdata, msg):
    global messages_count
    print(f"Received message from {msg.topic}: {msg.payload.decode()}")

    # Parse the received message
    data = json.loads(msg.payload.decode())

    # Insert the data into the database
    cursor.execute("INSERT INTO dataflow (data) VALUES (?)", [json.dumps(data)])
    conn.commit()
    print("Data stored in the database.")
    messages_count += 1
    print(f"Total messages stored: {messages_count}")
    if messages_count == 30:
        mcc = calculate_mcc()
        accuracy = calculate_accuracy()
        dictionary = {'mcc': mcc, 'accuracy': accuracy}
        client.publish(new_topic, json.dumps(dictionary))
        messages_count = 0

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