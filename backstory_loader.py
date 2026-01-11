# File to load backstory data
import os
import pandas as pd
from helper import Model, dummy_function


class DataSetLoader:
    def __init__(self, backstory_path):
        self.backstory_path = backstory_path
        self.model = Model()

    def generate_output(self):
        if not os.path.exists(self.backstory_path):
            print(f"Error: Backstory file not found at {self.backstory_path}")
            return ""

        # try:
        with open(self.backstory_path, 'r', encoding='utf-8') as f:
            print("file opened")
            df = pd.read_csv(self.backstory_path)
            df = df[["id", "book_name", "char", "content"]]

            results = []

            for idx, row in df.iterrows():
                id = str(row["id"])
                bookname = str(row["book_name"])
                char = str(row["char"])
                content = str(row["content"])

                verdict, reason = dummy_function(bookname, char, content, self.model)
                # expected: {"verdict": "...", "reason": "..."}

                results.append({
                    "story_id": id,
                    # "book_name": bookname,
                    # "char": char,
                    # "content": content,
                    "prediction": verdict,
                    "rationale": reason
                })
            out_df = pd.DataFrame(results)
            out_df.to_csv("results.csv", index=False, encoding="utf-8")
            print("Output saved to output.csv")
            
        # except Exception as e:
        #     print(f"Error reading backstory file {self.backstory_path}: {e}")
        #     return ""