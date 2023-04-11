from backend.storage.redis import RedisMemory

class ShortTermMemory:
    def __init__(self, redis_storage: RedisMemory):
        self.redis_storage = redis_storage

    def update(self, text):
        self.redis_storage.rpush('short_term_memory', text)

        # 只保留最近的10个文本
        self.redis_storage.ltrim('short_term_memory', -10, -1)

    def get_all_texts(self):
        return self.redis_storage.lrange('short_term_memory', 0, -1)