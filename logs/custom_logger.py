import logging

class CustomLogger:
    def __init__(self, name, log_file='logfile.log', level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.handler = logging.FileHandler(log_file)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
    
    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

# Example usage:
if __name__ == "__main__":
    custom_logger = CustomLogger(__name__)  # Use __name__ as the logger name
    custom_logger.log("This is a test log message")
