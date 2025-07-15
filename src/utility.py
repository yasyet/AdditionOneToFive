# --- Utility terminal commands ---
def choose_option(prompt, options):
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        choice = input("Enter the number of your choice: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please try again.")

def ask_two_numbers():
    while True:
        try:
            num1 = int(input("Enter the first number (1-5): "))
            num2 = int(input("Enter the second number (1-5): "))
            if 1 <= num1 <= 5 and 1 <= num2 <= 5:
                return num1, num2
            else:
                print("Numbers must be between 1 and 5. Please try again.")
        except ValueError:
            print("Invalid input. Please enter integers only.")

def clear_terminal():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def terminal_seperator():
    print("-" * 50)