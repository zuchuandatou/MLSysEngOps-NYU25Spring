# running on my laptop


from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
from gevent import monkey
import gevent

monkey.patch_all()
setup_logging("INFO", None)

# User behavior
class PredictUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        self.client.post("/predict", json={
            "user_id": 42,
            "sequence": [10, 11, 23, 99],
            "top_k": 5
        })

# Setup environment
env = Environment(user_classes=[PredictUser], host="http://129.114.27.220:8003")
env.create_local_runner()

# Optional: print stats in terminal
gevent.spawn(stats_printer(env.stats))
gevent.spawn(stats_history, env.runner)

# Start test
env.runner.start(user_count=10000, spawn_rate=100)
gevent.sleep(300)
env.runner.quit()
