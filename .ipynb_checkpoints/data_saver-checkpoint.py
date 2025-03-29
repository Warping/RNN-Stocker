import pickle
import os

class DataSaver:
    def __init__(self, path):
        self.path = path
        self.data = None

    def save(self, data):
        print(f"Saving data to {self.path}")
        self.data = data
        with open(self.path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        if self.data is not None:
            return self.data
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                self.data = pickle.load(f)
            return self.data
        # Check if folder exists
        folder = os.path.dirname(self.path)
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist")
            print("Creating folder...")
            os.makedirs(folder)
        print(f"File {self.path} does not exist")
        return None