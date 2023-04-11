import tensorflow as tf
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.python.keras.models import Model
from services.preprocess import text_to_sequence
import numpy as np

class LSTMModel:
    """
    LSTM模型，用于生成回答
    """
    def __init__(self):
        self.max_len = 50  # 模型输入序列的最大长度

        # 定义模型的输入层
        input_layer = Input(shape=(self.max_len,), dtype='int32')

        # 定义模型的嵌入层
        embedding_layer = Embedding(input_dim=10000, output_dim=128, input_length=self.max_len)(input_layer)

        # 定义模型的LSTM层
        lstm_layer = LSTM(128)(embedding_layer)

        # 定义模型的输出层
        output_layer = Dense(1, activation='sigmoid')(lstm_layer)

        # 定义模型
        self.model = Model(inputs=input_layer, outputs=output_layer)

        # 编译模型
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X, y):
        """
        训练模型
        """
        self.model.fit(X, y, batch_size=128, epochs=10, validation_split=0.2)

    def predict(self, X):
        """
        预测回答
        """
        y_pred = self.model.predict(X)
        return y_pred

    def generate_output(self, input_text):
        """
        根据输入文本生成回答
        """
        # 将输入文本转换为模型的输入序列
        input_seq = self.text_to_sequence(input_text)

        # 使用模型预测回答的概率
        output_prob = self.predict(input_seq)

        # 将概率转换为回答文本
        output_text = self.prob_to_text(output_prob)

        return output_text

    def text_to_sequence(self, text):
        """
        将文本转换为模型的输入序列
        """
        # 文本预处理和序列化操作
        input_seq = text_to_sequence(text, self.word_dict, self.max_seq_len)
        return input_seq


    def prob_to_text(self, prob):
        """
        将概率转换为回答文本
        """
        # 将概率序列转换为单词序列
        word_seq = [self.idx_to_word[idx] for idx in np.argmax(prob, axis=-1)]

        # 将单词序列转换为文本
        output_text = ' '.join(word_seq)

        return output_text