import polars as pl

class PolarsStorage:
    def init(self):
        self.df = pl.DataFrame({'question': [], 'answer': []})

    def add_row(self, question, answer):
        self.df = self.df.with_columns([
            pl.Series('question', [question]),
            pl.Series('answer', [answer])
        ])

    def get_all_rows(self):
        return self.df.to_arrow()

    def delete_all_rows(self):
        self.df = pl.DataFrame({'question': [], 'answer': []})