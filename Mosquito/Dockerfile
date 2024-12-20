# Use the official Eclipse Mosquitto image as the base
FROM eclipse-mosquitto:latest

# Copy a custom Mosquitto configuration file into the container
COPY mosquitto.conf /mosquitto/config/mosquitto.conf

# Expose ports
EXPOSE 1883
EXPOSE 9001

# Start Mosquitto when the container runs
CMD ["/usr/sbin/mosquitto", "-c", "/mosquitto/config/mosquitto.conf"]