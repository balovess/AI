import jieba

# 停用词列表，可以根据具体需求进行修改和扩展
stopwords = ['的', '了', '是', '我', '你', '他', '她', '我们', '你们', '他们', '她们']


class PreprocessService:
    def init(self):
        pass

    def preprocess(self, text):
        
        # 使用jieba进行分词
        words = jieba.lcut(text)

        # 过滤掉停用词和标点符号
        stopwords = ['的', '了', '呢', '吗', '吧', '啊', '呀']
        words = [word for word in words if word not in stopwords and word.isalnum()]

        # 将分词结果拼接成一个字符串
        preprocessed_text = ' '.join(words)

        return preprocessed_text

    def preprocess_sensory_memory(text):
       
        # 分词
        words = jieba.lcut(text)
        # 去除停用词
        words = [w for w in words if w not in stopwords]
        # 返回处理后的文本
        return ' '.join(words)

    def preprocess_short_term_memory(text):
        """
        针对短时记忆的文本预处理函数
        """
        # TODO: 短时记忆的文本预处理操作
        return text

    def preprocess_long_term_memory(text):
        """
        针对长时记忆的文本预处理函数
        """
        # TODO: 长时记忆的文本预处理操作
        return text



    def text_to_sequence(text, word_dict, max_seq_len):
        """
        将文本转换为数字序列
        """
        # 分词并去除停用词
        words = [w for w in jieba.cut(text) if w not in stopwords]
        # 将词语映射为数字
        sequence = [word_dict.get(w, 0) for w in words]
        # 截断或填充序列，使其长度为max_seq_len
        sequence = sequence[:max_seq_len] + [0] * (max_seq_len - len(sequence))
        # 返回序列
        return sequence