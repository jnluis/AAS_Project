import sqlite3
import json
import paho.mqtt.client as mqtt
import os
import pandas as pd

# Set up SQLite connection
db_dir = "/scripts/db"
db_file = os.path.join(db_dir, "mqtt_data.db")

# Create the database directory if it doesn't exist
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create the table for storing MCC values if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS mcc_data (
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        mcc_value REAL,
        accuracy REAL
    )
""")
conn.commit()
conn.close()


broker = "mosquitto"  # Change this to the correct broker IP or hostname if needed
port = 12000  # MQTT port

# Callback when a message is received
def on_message(client, userdata, msg):
    print(f"Received message from {msg.topic}: {msg.payload.decode()}")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Parse the received message (assuming it has an 'mcc' key)
    try:
        dict_data = json.loads(msg.payload.decode())
        mcc_value = dict_data.get("mcc")
        accuracy = dict_data.get("accuracy")
        if mcc_value is not None:
            # Insert the MCC value into the database
            cursor.execute("INSERT INTO mcc_data (mcc_value,accuracy) VALUES (?,?)", [mcc_value,accuracy])
            conn.commit()
            print("MCC value stored in the database.")
            conn.close()

            new_topic = "retrain"
            # Infinite loop to send the message every 30 seconds
            # retrain = {"retrain": True}
            # message = pd.Series(retrain).to_json()
            # client.publish(new_topic, message)
            # print(f"Published to {new_topic}: {message}")

            # #Thresold value
            if mcc_value < 0.94:
                new_topic = "retrain"
                # Infinite loop to send the message every 30 seconds
                retrain = {"retrain": True}
                message = pd.Series(retrain).to_json()
                client.publish(new_topic, message)
                print(f"Published to {new_topic}: {message}")

                print("MCC value is less than 0.65")

            else:
                print("MCC value is greater than 0.65")
        else:
            print("No MCC value found in the message.")
    except json.JSONDecodeError:
        print("Error decoding message.")

# Create an MQTT client instance
client = mqtt.Client(protocol=mqtt.MQTTv311)

# Assign the on_message callback
client.on_message = on_message

# Connect to the broker
client.connect(broker, port)

# Subscribe to the topic where messages will be received (replace with the correct topic)
topic = "grafana"  # Replace with your topic
client.subscribe(topic)
print(f"Subscribed to {topic} and waiting for messages...")

# Start the MQTT client loop to process received messages
client.loop_forever()



    