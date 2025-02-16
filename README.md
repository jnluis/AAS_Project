# AAS - Machine Learning Applied to Security

### Academic Year: 2024/25

### Grade: 19.25

To start Grafana and MQTT:

```shell
sudo docker compose up --build -d
```

The default credentials for grafana are admin for both user and password.

In Grafana go to "Connections"->"Add new connection" and search for sqlite.

Install sqllite plugin, after click on "Add new data source"

Specify the path to the Grafana's database each is "/scripts/db/mqtt_data.db" in the Path field and do Save & Test.

It will show "Data Source is working" and then click on "building a dashboard".

There go to "+ Add Visualization" and add the frser-sqlite-datasource(default). In the query space add the query:
```sql
SELECT
    strftime('%s', timestamp) AS time,
    mcc_value,
    accuracy
FROM
    mcc_data
ORDER BY
    time ASC  
```

To start the Publisher:
```shell
cd Gerador_Dados
python3 publisher.py
```

To start the Consumer:
```shell
cd Model_Retrainer
python3 consumer.py
```

To start the Model Retrain:
```shell
cd Model_Retrainer
python3 retrain_model.py
```

Project done in collaboration with [@Diogo Almeida](https://github.com/twisteddi84).