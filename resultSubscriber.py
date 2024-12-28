import paho.mqtt.client as mqtt

broker = "localhost"
port = 1883
topic = "resultsflow"

def on_message(client, userdata, message):
    print(f"Received Result: {message.payload.decode()}")

client = mqtt.Client()
client.connect(broker, port)
client.subscribe(topic)
client.on_message = on_message
print(f"Subscribed to topic '{topic}'")
client.loop_forever()
