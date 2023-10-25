 

import math
import time


def range_from_timestamps(start_timestamp: int) -> str:
    current_timestamp = int(time.time() * 1000)  # Current time in milliseconds
    range_seconds = (current_timestamp - start_timestamp) / 1000  # Convert milliseconds to seconds
    range_seconds_rounded = math.floor(range_seconds) if range_seconds > 0 else 1  # Round to the nearest integer, ensuring it's at least 1s
    range_str = f'{range_seconds_rounded}s'
    return range_str