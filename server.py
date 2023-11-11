import os
import uuid
import traceback
import flwr as fl
import requests
from flask import Flask, request, jsonify
from flask_cors.extension import CORS
from flask_restful import Resource, Api
import logging
from threading import Thread
from experiment.experiment import Experiment

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module

app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

clients = os.environ['CLIENTS'].split(',')
logger.info("Clients: %s", clients)

class StartFL(Resource):
    def get(self):
        job_id = str(uuid.uuid4())  # Generate a unique job ID

        # Create and configure a new experiment
        experiment = Experiment()
        
        server_thread = Thread(target=start_fl_server,args=(job_id, experiment.strategy, experiment.rounds), daemon=True)
        server_thread.start()
        
        # Wait for the server to start up

        server_thread.join(timeout=5)

        # Send start signal to clients
        for client in clients:
            try:
                r = requests.get(f"http://{client}/client_api/start-fl-client")
                logger.info("Response from client %s: %s", client, r.text)
            except Exception as e:
                exception_traceback = traceback.format_exc()
                print(f"Error sending start signal to client {client}: {e}\n{exception_traceback}")
                return {"status": "error", "message": str(e), "trace": exception_traceback}, 500

        response = {"status": "started", "job_id": job_id, "experiment_name": experiment.name}

        return response, 200


def start_fl_server(job_id, strategy,rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )        
        
    except Exception as e:
        logger.info(f"Server stopped with message: {str(e)}")


class Ping(Resource):
    def get(self):
        logger.info("Received ping. I'm server and I'm alive.")
        return 'I am server and I am alive', 200
    

api.add_resource(Ping, '/api/ping')
api.add_resource(StartFL, '/api/start-fed-learning')


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT"))
    app.run(debug=True, threaded=True, host="0.0.0.0", port=port)
