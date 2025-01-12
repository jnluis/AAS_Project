import paho.mqtt.client as mqtt
import pandas as pd
import time

# Load the data
train_df = pd.read_csv("../Data_files/test_data.csv")

# Define broker details
broker = "localhost"
port = 12000
topic = "dataflow"

# Create an MQTT client instance
client = mqtt.Client(protocol=mqtt.MQTTv311)

# Connect to the broker
client.connect(broker, port)

# Publish the data
for index, row in train_df.iterrows():
    message = row.to_json()
    client.publish(topic, message)
    print(f"Published to {topic}: {message}")
    # Wait for 15 seconds before sending the next row
    time.sleep(1)

# Disconnect the client
client.disconnect()



