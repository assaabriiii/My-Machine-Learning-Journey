import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load your trained brain
model = load_model('rps_model.h5')

# The mapping (Check your training output to be sure of the order!)
# Usually alphabetical: Paper, Rock, Scissors
class_labels = ['Paper', 'Rock', 'Scissors']

# 2. Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Preprocess the frame for the AI
    # Resize to 150x150 (same as training)
    img = cv2.resize(frame, (150, 150))
    # Convert to array and normalize (0-1)
    img_array = np.array(img) / 255.0
    # Add the "Batch" dimension (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Predict
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    predicted_class = class_labels[index]
    confidence = prediction[0][index]

    # 5. Display Result
    # Draw a rectangle and text
    cv2.putText(frame, f"AI Sees: {predicted_class} ({int(confidence*100)}%)", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Rock Paper Scissors AI', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()