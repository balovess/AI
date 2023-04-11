import polars as pl

class PolarsMemory:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pl.DataFrame()

    def load(self):
        self.df = pl.read_csv(self.file_path)

    def save(self):
        self.df.to_csv(self.file_path)

    def append(self, row_dict):
        self.df = self.df.append(pl.Series(row_dict))

    def query(self, query_str):
        return self.df.lazy().filter(query_str).collect()