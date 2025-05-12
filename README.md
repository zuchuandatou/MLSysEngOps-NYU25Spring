### Data pipeline
#### Create a persistent storage on Chameleon
I have created one object store on Chameleon, to do this, I created and followed `1_create-server.ipynb` and `2_object.ipynb`. 
To summarize the steps I have taken:
* Created a server named `node-persist-project37`. 
* Created `object-persist-project37` under CHI@TACC, authenticate it from my compute instance
* Retrieved code on our team github repo, and performed the ELT pipeline using docker (under `docker/docker-compose-etl`)
* Mounted our object store to local file system

In our object store, under `/mnt/object`, it contains the MovieLens raw, training, testing, evaluation dataset of ~6GB.

I have created three block storage on Chameleon, to do this, I created and followed `3_block.ipynb`.
To summarize the steps I have taken:
* Created `block-persist-project37` under KVM@TACC, attached it to `node-persist-project37`
* Created docker volumes on persistent storage
  * `/mnt/block/postgres_data` contains the user info (userId and itemId)
  * `/mnt/block/minio_data` contains model artifacts
  * `mlflow` contains experiment artifacts

##### Offline Data:
- `ml-192m/` → which stores MovieLens 192M raw dataset and contains the raw ratings.csv
- `training/` → Cleaned-up data
- `models/` → Trained models
- `logs/` → User activity logs
##### Online Data: 
- Simulate user interaction with movie data
##### Data Pipeline:
[Data Sources] → [Ingestion] → [Cleaning] → [Feature Store]
##### Difficulty Point: Interactive Dashboard
- Use Grafana to visualize user activity pattern and recommendation performance

