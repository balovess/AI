# 在 input.py 中，应包含以下业务逻辑：
# 接收前端传来的用户输入数据，并将数据存储到 Redis 数据库中的感时记忆中。
# 对用户输入数据进行初步处理，例如去除无用词汇、标点符号等，将处理后的结果存储到短时记忆中。
# 对短时记忆中的数据进行深度处理，例如提取实体、关键词等信息，将处理后的结果存储到长时记忆中。
# 根据业务需求，实现相应的文本预处理逻辑，例如分词、词性标注、命名实体识别等。
# 实现动态训练模型的逻辑，从数据库中读取数据进行训练，并将训练后的模型存储到相应的位置。
# 实现数据清理和维护逻辑，例如定期清理过期数据，更新模型等。

from flask import Blueprint, request
from storage.redis import RedisMemory
from storage.short_term_memory import ShortTermMemory
from storage.sensory_memory import SensoryMemory
from preprocess import preprocess_sensory_memory

input_service = Blueprint('input_service', __name__)

redis_db = RedisMemory()
short_term_memory = ShortTermMemory()
sensory_memory = SensoryMemory()

@input_service.route('/api/input', methods=['POST'])
def handle_input():
    # 从请求中获取用户输入的数据
    user_input = request.json['input']

    # 对用户输入的文本进行预处理
    preprocessed_input = preprocess_sensory_memory(user_input)

    # 将预处理后的文本存储到感时记忆中
    sensory_memory.add(preprocessed_input)

    # 对感时记忆中的文本进行初步处理，并存储到短时记忆中
    short_term_memory.add(preprocessed_input)

    # 将用户输入数据存储到Redis数据库中，以备后续使用
    redis_db.set('user_input', user_input)

    # 返回一个空的响应，表示处理成功
    return '', 204