docker build -t custom-mosquitto .

docker run -d --name mosquitto -p 12000:12000 -p 9001:9001 custom-mosquitto