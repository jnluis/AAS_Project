import paho.mqtt.client as mqtt
import sqlite3
import json

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

# Broker details
broker = "localhost"
port = 12000
topic = "dataflow"

# Callback when a message is received
def on_message(client, userdata, msg):
    print(f"Received message from {msg.topic}: {msg.payload.decode()}")

    # Parse the received message
    data = json.loads(msg.payload.decode())

    # Insert the data into the database
    cursor.execute("INSERT INTO dataflow (data) VALUES (?)", [json.dumps(data)])
    conn.commit()
    print("Data stored in the database.")

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