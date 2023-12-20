import tkinter
from keras.models import load_model
import numpy as np

class MyGUI:
    def __init__(self):
        self.root = tkinter.Tk(className=" IRIS Flower Predictor")

        self.root.geometry("400x550")

        self.frame = tkinter.Frame(self.root)

        self.frame.pack(pady=20, padx=60, fill="both")

        self.label = tkinter.Label(self.root, text="IRIS Flower Predictor",  font=("Roboto", 20))
        self.label.pack(pady=5, padx=10)
        
        #label
        self.label = tkinter.Label(self.root, text="sepal_length",  font=("Roboto", 10))
        self.label.pack(pady=5, padx=10)

        self.entry1 = tkinter.Entry(self.root)
        self.entry1.pack(pady=2, padx=10)

        #label
        self.label = tkinter.Label(self.root, text="sepal_width",  font=("Roboto", 10))
        self.label.pack(pady=5, padx=10)

        self.entry2 = tkinter.Entry(self.root)
        self.entry2.pack(pady=2, padx=10)

        #label
        self.label = tkinter.Label(self.root, text="petal_length",  font=("Roboto", 10))
        self.label.pack(pady=5, padx=10)

        self.entry3 = tkinter.Entry(self.root)
        self.entry3.pack(pady=2, padx=10)

        #label
        self.label = tkinter.Label(self.root, text="petal_width",  font=("Roboto", 10))
        self.label.pack(pady=5, padx=10)

        self.entry4 = tkinter.Entry(self.root)
        self.entry4.pack(pady=2, padx=10)

        self.button = tkinter.Button(self.root, text="Submit", command=self.submit)
        self.button.pack(pady=12, padx=10)

        self.root.mainloop()

    def submit(self):
        try:
            sepal_length = float(self.entry1.get())
            sepal_width = float(self.entry2.get())
            petal_length = float(self.entry3.get())
            petal_width = float(self.entry4.get())
        except ValueError:
        # Handle the case where the input is not a valid float
            print("Invalid input. Please enter numeric values.")
            return
        
        array = np.array([sepal_length, sepal_width, petal_length, petal_width], dtype=np.float32)

        model = load_model("models\IRIS_prediction_fix.h5")
        yhat = model.predict(np.expand_dims(array, 0))

        prediction = np.argmax(yhat, axis=-1)
        
        print("Array: ", array)
        print("IRIS - ", yhat)
        print("Predicted probabilities:", prediction)

        if prediction == [0]:
            label_probabilities = tkinter.Label(self.root, text=f"Predicted probabilities\n{yhat}\n\nIt's IRIS-SETOSA" ,  font=("Roboto", 10))
            label_probabilities.pack(pady=5, padx=10)

        elif prediction == [1]:
            label_probabilities = tkinter.Label(self.root, text=f"Predicted probabilities\n{yhat}\n\nIt's IRIS-VERSICOLOR" ,  font=("Roboto", 10))
            label_probabilities.pack(pady=5, padx=10)

        elif prediction == [2]:
            label_probabilities = tkinter.Label(self.root, text=f"Predicted probabilities\n{yhat}\n\nIt's IRIS-VIRGINICA" ,  font=("Roboto", 10))
            label_probabilities.pack(pady=5, padx=10)
MyGUI()



