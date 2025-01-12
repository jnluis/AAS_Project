import paho.mqtt.client as mqtt
import pandas as pd
import time

# Define broker details
broker = "localhost"
port = 12000
topic = "retrain"

# Create an MQTT client instance
client = mqtt.Client(protocol=mqtt.MQTTv311)

# Connect to the broker
client.connect(broker, port)

# Infinite loop to send the message every 30 seconds
retrain = {"retrain": True}
time.sleep(15)
while True:
    # Convert message to JSON
    message = pd.Series(retrain).to_json()

    # Publish the message
    client.publish(topic, message)
    print(f"Published to {topic}: {message}")
    # Wait for 30 seconds
    time.sleep(45)

# Disconnect the client (this will never be reached due to the infinite loop)
client.disconnect()