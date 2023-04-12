import redis

class RedisStorage:
    
    def __init__(self, host='localhost', port=6379, db=0):
        # 连接到Redis数据库
        self.redis = redis.StrictRedis(host=host, port=port, db=db)

    def set(self, key, value, expire=None):
        self.redis.set(key, value)
        if expire:
            self.redis.expire(key, expire)

    def get(self, key):
        # 获取键对应的值
        return self.redis.get(key)

    def delete(self, key):
        # 删除键值对
        self.redis.delete(key)

    def lpush(self, key, value):
        # 向列表左侧添加元素
        self.redis.lpush(key, value)

    def rpush(self, key, value):
        # 向列表右侧添加元素
        self.redis.rpush(key, value)

    def lrange(self, key, start, end):
        # 获取列表中指定范围的元素
        return self.redis.lrange(key, start, end)

    def ltrim(self, key, start, end):
        # 修剪列表，仅保留指定范围内的元素
        self.redis.ltrim(key, start, end)