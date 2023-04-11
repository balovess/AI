from backend.storage.redis import RedisStorage

class SensoryMemory:
    def __init__(self, redis_storage: RedisStorage):
        self.redis_storage = redis_storage

    def append(self, text):
        self.redis_storage.rpush('sensory_memory', text)

    def get_all_texts(self):
        return self.redis_storage.lrange('sensory_memory', 0, -1)