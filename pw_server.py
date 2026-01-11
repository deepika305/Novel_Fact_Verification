import pathway as pw
import csv
import json
import os
from helper import dummy_function, Model

class InputSchema(pw.Schema):
    id: int
    book_name: str
    char: str
    content: str

@pw.udf
def run_check(book_name: str, char: str, content: str) -> str:
    try:
        # Initialize model inside UDF for worker process compatibility
        model = Model()
        verdict, reason = dummy_function(book_name, char, content, model)
        return json.dumps({"verdict": verdict, "reason": reason})
    except Exception as e:
        return json.dumps({"verdict": "Error", "reason": str(e)})

def on_change(key, row, time, is_addition):
    if is_addition:
        try:
            res = json.loads(row['result'])
            with open("output.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([res['verdict'], res['reason']])
                f.flush()
        except Exception as e:
            print(f"Error writing to output CSV: {e}")

def main():
    # Initialize output file with headers
    with open("output.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["verdict", "reason"])

    # Input file path
    input_path = "Dataset/input.csv"
    
    # Read the input CSV in streaming mode
    # set primary_key to 'id' so that Pathway uses the 'id' column from CSV as its internal id
    table = pw.io.csv.read(input_path, schema=InputSchema, mode="streaming")
    
    # Apply the UDF
    result_table = table.select(
        book_name=pw.this.book_name,
        char=pw.this.char,
        content=pw.this.content,
        result=run_check(pw.this.book_name, pw.this.char, pw.this.content)
    )
    
    # Subscribe to write output on changes
    pw.io.subscribe(result_table, on_change)
    
    # Run the pipeline
    print(f"Starting Pathway pipeline... Watching {input_path}")
    pw.run()

if __name__ == "__main__":
    print("Starting Novel Fact Verification Process...")
    main()
