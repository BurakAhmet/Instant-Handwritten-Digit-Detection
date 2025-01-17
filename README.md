# Handwritten Digit Detection
This project implements a handwritten digit detection system using OpenCV and TensorFlow's convolutional neural network (CNN) model. The goal of the project is to allow users to draw digits on a canvas, after which the program detects and recognizes the digits, displaying the result alongside the original drawing. 

## Preview
https://github.com/BurakAhmet/Instant-Handwritten-Digit-Detection/assets/89780902/571a6ae5-f02f-4dc8-bac0-5a0b571aa0e9

## Model

**Final validation loss: 0.0144**

**Final validation accuracy: 0.9960**

For more details you can check [model_training.ipynb](https://github.com/BurakAhmet/Instant-Handwritten-Digit-Detection/blob/main/model_training.ipynb) file.

## Usage
1- Run the program.

2- Draw digits on the canvas using the left mouse button.

3- Release the mouse button to signal drawing completion.

4- View the recognized digits alongside the original drawing.

5- Press "q" to quit the program or "r" to clear the canvas.

## Technologies Used

* Python: The project is developed using Python programming language.
* TensorFlow: Used for Creating the model, training the data and making predictions.
* OpenCV: Used for image preprocessing, visualization of results, user interface.
* NumPy: Employed for array manipulation and normalization of input data.
* Google Colab: Used for fast model training with GPUs.
