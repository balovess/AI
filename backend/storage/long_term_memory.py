import pandas as pd
import polars as pl
from backend.storage.polars import PolarsStorage

class LongTermMemory:
    def __init__(self, polars_storage: PolarsStorage):
        self.polars_storage = polars_storage
        self.memory = self.polars_storage.load_dataframe('long_term_memory')

    def update(self, text):
        # 将新文本添加为新行
        new_row = pd.DataFrame({'text': [text], 'vector': [None]})
        self.memory = self.memory.append(new_row)
                # 计算每个文本的向量表示
        embeddings = self.polars_storage.get_embeddings(self.memory['text'])
        self.memory['vector'] = embeddings

        # 将更新后的数据帧保存到 Polars 存储中
        self.polars_storage.save_dataframe(self.memory, 'long_term_memory')