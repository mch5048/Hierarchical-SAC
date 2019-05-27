from functools import wraps
import threading
import time


last_called = dict()  # When last called, and with what result

def rate_limited(seconds=10, timer=time.time):
    def decorator(func):
        last_called[func] = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = timer()
            call_data = last_called.get(func, None)
            if call_data is None or now - call_data[0] >= seconds:
                result = func(*args, **kwargs)
                last_called[func] = (now, result)
            else:
                result = call_data[1]  # Replay rate-limited result
            return result
        return wrapper
    return decorator