print("Welcome to the Novel Fact Verification Main file!")
# show the list of datasets to be used and the novel to be processed
# give user option to select dataset and novel
# load the selected novel and dataset
from preprocessor_helper import NovelPreprocessor
from backstory_loader import DataSetLoader
import os

if __name__ == "__main__":
    print("Hope you have run the preprocessor.py file to create FAISS index for the novel.")
    print("Starting Novel Fact Verification Process...")
    # specify novel path and name
    # show the list of files in dataset folder
    path = "Dataset/new.csv"
    dataset_loader = DataSetLoader(backstory_path=path)
    dataset_loader.generate_output()