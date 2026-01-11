print("Welcome to the Novel Fact Verification Main file!")
# show the list of datasets to be used and the novel to be processed
# give user option to select dataset and novel
# load the selected novel and dataset
from preprocessor_helper import NovelPreprocessor
from backstory_loader import DataSetLoader
import os

def select_file(folder_path="Dataset"):
    """
    Lists all CSV files in the specified folder, asks the user to select one
    by index, and returns the full path of the selected file.
    """
    
    # 1. Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return None

    # 2. Get a list of all .csv files in the folder
    # We use .lower() to ensure we catch '.CSV' as well as '.csv'
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

    # 3. Check if any CSV files were found
    if not files:
        print(f"No CSV files found inside the '{folder_path}' folder.")
        return None

    # 4. Display the files with numbers
    print(f"\n--- CSV Files found in '{folder_path}' ---")
    for index, file_name in enumerate(files, 1):
        print(f"{index}. {file_name}")
    print("---------------------------------------")

    # 5. Loop until the user provides valid input
    while True:
        try:
            user_input = input("Enter the number of the file you want to select: ")
            selection = int(user_input)

            # Check if the number is within the list range
            if 1 <= selection <= len(files):
                # Adjust for 0-based indexing used by Python lists
                selected_file_name = files[selection - 1]
                
                # Create the full cross-platform path
                full_path = os.path.join(folder_path, selected_file_name)
                print(f"Selected: {selected_file_name}")
                return full_path
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(files)}.")
        
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

if __name__ == "__main__":
    print("Hope you have run the preprocessor.py file to create FAISS index for the novel.")
    print("Starting Novel Fact Verification Process...")
    # specify novel path and name
    # show the list of files in dataset folder
    path = select_file()
    print(f"Path selected is {path}")
    dataset_loader = DataSetLoader(backstory_path=path)
    dataset_loader.generate_output()