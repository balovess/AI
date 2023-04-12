import json

class InputService:
    def init(self, preprocess_service, sensory_memory, short_term_memory, long_term_memory):
        self.preprocess_service = preprocess_service
        self.sensory_memory = sensory_memory
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory

    def handle_input(self, request_data):
        # 解析输入数据
        input_text = request_data.get('text')

        # 进行感时记忆储存前预处理
        preprocessed_text = self.preprocess_service.preprocess_sensory_memory(input_text)

        # 存储到感时记忆
        self.sensory_memory.add(preprocessed_text)

        # 分类并存储到短时记忆或长时记忆
        if self.short_term_memory.classify(preprocessed_text):
            self.short_term_memory.add(preprocessed_text)
        else:
            self.long_term_memory.add(preprocessed_text)

        # 返回成功响应
        response_data = {'success': True}
        return json.dumps(response_data)