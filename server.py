import os
import uuid
import traceback
import flwr as fl
import requests
from flask import Flask
from flask_cors.extension import CORS
from flask_restful import Resource, Api
import logging
from threading import Thread
from experiment.experiment import Experiment
from utils.simple_utils import parse_docker_compose

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App and API Configuration
app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Function to Start Federated Learning Server
def start_fl_server(job_id, strategy, rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"FL Server error: {e}", exc_info=True)

# Resource to Handle Starting of FL Process
class StartFL(Resource):
    def get(self):
        job_id = str(uuid.uuid4())
        experiment = Experiment()

        server_thread = Thread(target=start_fl_server, args=(job_id, experiment.strategy, experiment.rounds), daemon=True)
        server_thread.start()
        server_thread.join(timeout=5)

        clients = parse_docker_compose("docker-compose.yml")
        for client in clients:
            try:
                response = requests.get(f"http://{client}/client_api/start-fl-client")
                logger.info(f"Response from client {client}: {response.text}")
            except Exception as e:
                logger.error(f"Error with client {client}: {e}", exc_info=True)
                return {"status": "error", "message": str(e), "trace": traceback.format_exc()}, 500

        return {"status": "started", "job_id": job_id, "experiment_name": experiment.name}, 200

# Resource for Health Check
class Ping(Resource):
    def get(self):
        logger.info("Ping received.")
        return 'Server is alive', 200

# Add Resources to API
api.add_resource(Ping, '/api/ping')
api.add_resource(StartFL, '/api/start-fed-learning')

# Main Function
if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 6000))
    app.run(debug=True, threaded=True, host="0.0.0.0", port=port)
