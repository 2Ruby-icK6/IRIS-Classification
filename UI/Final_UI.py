import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model = keras.models.load_model('models/IRIS_prediction.h5')

# Load the training data to fit StandardScaler
df_train = pd.read_csv('IRIS.csv')
X_train_fit = df_train.drop(columns=['species'])
scaler = StandardScaler()
scaler.fit(X_train_fit)

# Function to handle prediction
def predict_species():
    try:
        # Get input data from the entry widgets
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())

        # Preprocess the input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_data = scaler.transform(input_data)

        # Make a prediction using the model
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)

        # Get the species name
        species_name = ["IRIS - Setosa", "IRIS - Versicolor", "IRIS - Virginica"]
        predicted_species = species_name[predicted_class]

        # Display the prediction
        result_label.config(text=f"Predicted probabilities\n{prediction}\n\nPredicted Species: {predicted_species}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
app = tk.Tk()
app.title("Iris Species Prediction")

# Create and place widgets
labels = ["Sepal Length:", "Sepal Width:", "Petal Length:", "Petal Width:"]
entries = []

for i, label_text in enumerate(labels):
    label = ttk.Label(app, text=label_text, font=('Roboto', 40))
    label.grid(column=0, row=i, padx=10, pady=10)

    entry = ttk.Entry(app, font=('Roboto', 40))
    entry.grid(column=1, row=i, padx=10, pady=10)
    entries.append(entry)

# Assigning the entry widgets to variable names
entry_sepal_length, entry_sepal_width, entry_petal_length, entry_petal_width = entries

predict_button = ttk.Button(app, text="Predict", command=predict_species)
predict_button.grid(column=0, row=len(labels), columnspan=2, pady=10)

result_label = ttk.Label(app, text="", font=('Roboto', 30))
result_label.grid(column=0, row=len(labels)+1, columnspan=2, pady=10)

# Run the main loop
app.mainloop()
