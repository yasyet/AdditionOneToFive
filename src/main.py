def main():
    # --- Import necessary modules ---
    from model import NeuralNetwork
    from layer import DenseLayer
    from activation_functions import sigmoid, relu, linear

    import utility as util
    import preprocess as prep
    import os
    import numpy as np

    # --- Program start ---
    util.clear_terminal()
    util.terminal_seperator()

    # --- Ask user what options they want to use ---
    print("Welcome to the Neural Network Addition Project!")
    util.terminal_seperator()

    selected_model_option = util.choose_option(
        "Choose an option:",
        ["Train a new model", "Load an existing model", "Exit"]
    )

    if selected_model_option == "Exit":
        print("Exiting the project. Goodbye!")
        os._exit(0)

    if selected_model_option == "Train a new model":
        selected_data_option = util.choose_option(
        "Choose an option:",
        ["Use existing data", "Generate synthetic data", "Exit"]
        )

        if selected_data_option == "Exit":
            print("Exiting the project. Goodbye!")
            os._exit(0)

    # --- Proceed with the selected options ---
    util.terminal_seperator()
    if selected_model_option == "Train a new model":
        print(f"You selected: {selected_model_option} and {selected_data_option}")
    else:
        print(f"You selected: {selected_model_option}")
    util.terminal_seperator()

    # --- Load or generate data based on user choice ---
    if selected_model_option == "Train a new model":
        if selected_data_option == "Use existing data":
            data = prep.load_data("data.txt", "data/")
        elif selected_data_option == "Generate synthetic data":
            data = prep.generate_data()
            save_data_option = util.choose_option(
                "Data generated successfully. Do you want to save it?",
                ["Yes", "No"]
            )
        if save_data_option == "Yes":
            prep.save_data(data)
            print("Data saved successfully.")
        else:
            print("Invalid option selected.")
            os._exit(1)

    # --- Continue with model training or evaluation ---
    if selected_model_option == "Train a new model":
        model = NeuralNetwork(layers=[
            DenseLayer(2, 4, activation_function=relu),
            DenseLayer(4, 1, activation_function=linear)
        ])
        model.train(data, epochs=10000, learning_rate=0.01)

        # --- Ask user if they want to evaluate the model ---
        evaluate_model_option = util.choose_option(
            "Training complete. Do you want to evaluate the model?",
            ["Yes", "No"]
        )

        if evaluate_model_option == "Yes":
            # Preprocess data for evaluation
            test_data = np.array(data)  # Convert to NumPy array
            inputs = test_data[:, :2]  # First two columns as inputs
            targets = test_data[:, 2:]  # Last column as targets
            evaluate_result = model.evaluate((inputs, targets))  # Pass as a tuple
            print(f"Evaluation result: {evaluate_result}")

        # --- Ask user if they want to save the model ---
        save_model_option = util.choose_option(
            "Do you want to save the trained model?",
            ["Yes", "No"]
        )

        if save_model_option == "Yes":
            model.save("model.pth")
            print("Model saved successfully.")

        # --- Ask user if they want to predict with the model ---
        predict_model_option = util.choose_option(
            "Do you want to make predictions with the trained model?",
            ["Yes", "No"]
        )

        if predict_model_option == "Yes":
            inputs = util.ask_two_numbers()
            prediction = model.predict(np.array([inputs]))
            print(f"Prediction for inputs {inputs}: {prediction}")
    elif selected_model_option == "Load an existing model":
        model = NeuralNetwork()
        model = model.load("model.pth")
        if model is None:
            print("Model not found. Please train a model first.")
            os._exit(1)
        print("Model loaded successfully.")
        util.terminal_seperator()

        inputs = util.ask_two_numbers()
        prediction = model.predict(np.array([inputs]))
        print(f"Prediction for inputs {inputs}: {round(prediction)}")

if __name__ == "__main__":
    main()