class DataIngestion:
    def __init__(self, input_path="data/raw/data.csv", output_path="data/processed/data.csv"):
        self.input_path = input_path
        self.output_path = output_path

    def load_and_store(self):
        df = pd.read_csv(self.input_path)
        df.to_csv(self.output_path, index=False)
        return df
