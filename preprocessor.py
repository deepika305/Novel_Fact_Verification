import os
from preprocessor_helper import NovelPreprocessor


def select_book(dataset_dir="Dataset/Books"):
    """
    Shows available books with option numbers.

    """
    
    # Check if books directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    # List only files (ignore folders)
    books = [
        book for book in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, book))
    ]

    if not books:
        raise ValueError("No books found in the books directory.")

    # Display books as numbered options
    print("\nAvailable Books:\n")
    for idx, book in enumerate(books, start=1):
        print(f"{idx}. {book}")

    # Take user input
    while True:
        try:
            choice = int(input("\nEnter the option number: "))
            if 1 <= choice <= len(books):
                selected_book = books[choice - 1]
                selected_book_path = os.path.join(dataset_dir, selected_book)
                return selected_book[:-4], selected_book_path
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a number.")
    return None, None

if __name__ == "__main__":
    print("Welcome to preprocessor.py file!")
    book_name, book_path = select_book()
    print(f"Selected Book: {book_name}\nPath: {book_path}")
    confirmation = input("Do you want to proceed with this book? (y/n): ").strip().lower()
    if confirmation != 'y':
        book_path = None
    if book_path==None:
        print("No book selected. Exiting.")
    else:
        preprocessor = NovelPreprocessor(novel_path=book_path, novel_name=book_name)
        preprocessor.forward()

