import redis
from typing import List, Tuple

class ShortTermMemory:
    def __init__(self, redis_host: str, redis_port: int):
        # 连接Redis数据库
        self.redis = redis.Redis(host=redis_host, port=redis_port)

    def add_message(self, message: str) -> None:
        # 将消息添加到短时记忆的列表中
        self.redis.rpush('short_term_memory', message)

    def get_messages(self, num_messages: int) -> List[str]:
        messages = []
        for i in range(num_messages):
            # 从短时记忆的列表中获取指定数量的消息
            message = self.redis.lpop('short_term_memory')
            if message is not None:
                # 将字节字符串转换为字符串，并添加到消息列表中
                messages.append(message.decode('utf-8'))
            else:
                # 如果列表已经为空，则停止获取消息
                break
        return messages

    def clear(self) -> None:
        # 删除短时记忆的列表
        self.redis.delete('short_term_memory')