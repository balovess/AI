from flask import Flask
from backend.models.lstm import LSTMModel
from backend.storage.redis import RedisStorage
from backend.storage.polars import PolarsStorage
from backend.storage.short_term_memory import ShortTermMemory
from backend.storage.long_term_memory import LongTermMemory
from backend.storage.sensory_memory import SensoryMemory
from backend.services.input import InputService
from backend.services.output import OutputService
from backend.services.training import TrainingService
from backend.services.preprocess import PreprocessService


# 创建 Flask 应用程序实例
app = Flask(__name__)


# 初始化 Redis 和 Polars 数据库
redis_storage = RedisStorage()
polars_storage = PolarsStorage()


# 初始化 LSTM 模型
lstm_model = LSTMModel()


# 初始化记忆
sensory_memory = SensoryMemory(redis_storage)
short_term_memory = ShortTermMemory(redis_storage)
long_term_memory = LongTermMemory(polars_storage)


# 创建服务实例
input_service = InputService(preprocess_service=PreprocessService(),
                             sensory_memory=sensory_memory,
                             short_term_memory=short_term_memory,
                             long_term_memory=long_term_memory)

output_service = OutputService(lstm_model=lstm_model,
                               short_term_memory=short_term_memory,
                               long_term_memory=long_term_memory)

training_service = TrainingService(lstm_model=lstm_model,
                                   short_term_memory=short_term_memory,
                                   long_term_memory=long_term_memory)


# 创建 Flask 路由
@app.route('api/input', methods=['POST'])
def handle_input():
    return input_service.handle_input()


@app.route('api/output', methods=['POST'])
def handle_output():
    return output_service.handle_output()


@app.route('api/train', methods=['POST'])
def handle_train():
    return training_service.handle_train()


# 启动 Flask 应用程序
if __name__ == '__main__':
    app.run()