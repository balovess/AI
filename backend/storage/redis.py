import redis

class RedisMemory:
    def __init__(self, host='localhost', port=6379, password=None):
        self.redis_client = redis.Redis(host=host, port=port, password=password)

    def set(self, key, value):
        self.redis_client.set(key, value)

    def get(self, key):
        return self.redis_client.get(key)

    def delete(self, key):
        self.redis_client.delete(key)