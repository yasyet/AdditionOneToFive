
import os

# --- Generate synthetic data ---
def generate_data():
    """
    Generates a dataset of pairs of numbers and their sum.
    Each pair consists of two integers between 1 and 5, inclusive.
    """

    # --- Generates 1.000 pairs of numbers ---
    import random
    data = []
    for _ in range(1000):
        num1 = random.randint(1, 5)
        num2 = random.randint(1, 5)
        data.append((num1, num2, num1 + num2))
    return data

# --- Save and load data ---
def save_data(data, filename='data.txt', path='data/'):
    """
    Saves the generated data to a file.
    Each line in the file contains a pair of numbers and their sum.
    """
    with open(os.path.join(path, filename), 'w') as f:
        for num1, num2, result in data:
            f.write(f"{num1} {num2} {result}\n")

# --- Load data from file ---
def load_data(filename='data.txt', path='data/'):
    """
    Loads data from a file.
    Each line in the file should contain a pair of numbers and their sum.
    """
    data = []
    with open(os.path.join(path, filename), 'r') as f:
        for line in f:
            num1, num2, result = map(int, line.split())
            data.append((num1, num2, result))
    return data