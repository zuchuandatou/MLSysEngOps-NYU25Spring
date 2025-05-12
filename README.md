### Data pipeline

#### Create a persistent storage on Chameleon
I created object-persist-project37 server on on Chameleon
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

