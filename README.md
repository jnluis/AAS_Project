To start grafana:

sudo docker run --name grafana   -p 3000:3000   -v grafana_data:/bitnami/grafana   bitnami/grafana:latest

The default credentials are admin for both user and password.

To add a MQTT as a data source, we have to specify a connection, but for that mosquitto and grafana containers have to be on the same network.

```shell
sudo docker network create mqtt_network
sudo docker network connect mqtt_network mosquitto
sudo docker network connect mqtt_network grafana
sudo docker restart mosquitto
sudo docker restart grafana
```