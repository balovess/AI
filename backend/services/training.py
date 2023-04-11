import tensorflow as tf
import numpy as np
from backend.models.lstm import LSTMModel
from backend.storage.redis import RedisStorage
from backend.storage.polars import PolarsStorage
from backend.storage.short_term_memory import ShortTermMemory
from backend.storage.long_term_memory import LongTermMemory
from backend.storage.sensory_memory import SensoryMemory
from backend.services.preprocess import preprocess_text

class DynamicTraining:
    def __init__(self, lstm_model: LSTMModel, dataset, learning_rate=0.001, batch_size=32, epochs=1):
        self.lstm_model = lstm_model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.redis_storage = RedisStorage()
        self.polars_storage = PolarsStorage()
        self.short_term_memory = ShortTermMemory(self.redis_storage)
        self.long_term_memory = LongTermMemory(self.polars_storage)
        self.sensory_memory = SensoryMemory(self.redis_storage)

    def train(self, input_text):
        # 将输入文本转换为模型输入格式
        input_text = preprocess_text(input_text)
        input_seq = np.array(self.dataset.texts_to_sequences([input_text]))
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=self.lstm_model.max_input_length, padding='post')
        
        # 从短时记忆中获取之前存储的所有文本
        short_term_texts = self.short_term_memory.get_all_texts()
        
        # 将短时记忆中的文本和当前输入文本组成新的训练数据
        texts = short_term_texts + [input_text]
        new_dataset = tf.keras.preprocessing.text_dataset_from_texts(texts, batch_size=self.batch_size)
        
        # 训练模型
        self.lstm_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        self.lstm_model.fit(new_dataset, epochs=self.epochs)
        
        # 更新短时记忆
        self.short_term_memory.update(input_text)
        
        # 将输入文本存入感时记忆
        self.sensory_memory.append(input_text)
        
        # 将长时记忆中的所有文本和当前输入文本组成新的数据集
        long_term_texts = self.long_term_memory.get_all_texts()
        texts = long_term_texts + [input_text]
        new_dataset = tf.keras.preprocessing.text_dataset_from_texts(texts, batch_size=self.batch_size)
        
        # 更新长时记忆
        self.lstm_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        self.lstm_model.fit(new_dataset, epochs=self.epochs)
        self.long_term_memory.update(input_text)
        
        # 将更新后的模型保存到Redis中
        self.redis_storage.save_model(self.lstm_model)