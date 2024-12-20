docker build -t custom-mosquitto .

docker run -d --name mosquitto -p 1883:1883 -p 9001:9001 custom-mosquitto