import cv2
import numpy as np
from keras.models import load_model

# Load the trained digit recognition model (replace with your model path)
model = load_model("model/model.h5")
model.load_weights("model/model_weights.h5")

# Define canvas dimensions
canvas_width = 500
canvas_height = 500

# Create a blank canvas (black background)
canvas = np.zeros((canvas_height, canvas_width, 3), np.uint8)
canvas[:] = (0, 0, 0)  # Set background to black

# Initialize drawing variables
drawing = False
last_x, last_y = None, None

# Flag to indicate drawing completion
drawing_complete = False


# Mouse callback function for drawing on the canvas (white lines)
def draw_on_canvas(event, x, y, flags, param):
    global drawing, last_x, last_y, drawing_complete
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (last_x, last_y), (x, y), (255, 255, 255), 4)  # White lines
            last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        drawing_complete = True  # Signal drawing completion


# Set up the window and mouse callback
cv2.namedWindow("Digit Recognition Canvas")
cv2.setMouseCallback("Digit Recognition Canvas", draw_on_canvas)

is_quit = False
while True:
    # Display the canvas (always show the original canvas with digits)
    cv2.imshow("Digit Recognition Canvas", canvas)

    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or is_quit:
        break

    if drawing_complete:
        # Process the entire canvas after drawing is complete
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Thresholding to isolate white pixels (digits)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]  # Adjust threshold if needed

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a temporary canvas to draw rectangles
        temp_canvas = canvas.copy()

        # Process and draw rectangles on the temporary canvas
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 10:
                continue

            # Calculate the bounding box around the digit region with some padding
            x_new, y_new, w_new, h_new = x - 45, y - 45, w + 90, h + 90

            # Ensure bounding box coordinates stay within the canvas
            x_new = max(0, x_new)
            y_new = max(0, y_new)

            # Extract the digit image with padding
            digit_img = thresh[y_new:y_new + h_new, x_new:x_new + w_new]

            # Resize and normalize the digit image for prediction
            digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
            digit_img = digit_img.reshape(1, 28, 28, 1) / 255.0  # Normalize the input

            # Predict the digit using the model
            prediction = model.predict(digit_img)
            digit = np.argmax(prediction)

            # Draw the digit image on a separate window for DEBUGGING
            # cv2.imshow('Digit Image', digit_img.reshape(28, 28) * 255)  # Rescale to 0-255 range
            # cv2.waitKey(0)  # Wait indefinitely until any key is pressed
            # cv2.destroyWindow('Digit Image')  # Close the window after key press

            # Draw a red rectangle around the digit and display the prediction
            cv2.rectangle(temp_canvas, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(temp_canvas, str(digit), (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Digit Recognition Canvas", temp_canvas)

        # Wait for a key press OR until the user starts drawing again
        while not drawing:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                is_quit = True
                break  # Quit if "q" is pressed
            elif key == ord("r"):
                canvas[:] = (0, 0, 0)
                break

        drawing_complete = False  # Reset for the next digit

cv2.destroyAllWindows()
