# train_model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load preprocessed data
preprocessed_dir = 'preprocessed_data'
data = []
labels = []

for idx, gesture_folder in enumerate(os.listdir(preprocessed_dir)):
    gesture_path = os.path.join(preprocessed_dir, gesture_folder)
    for img_file in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_file)
        img = np.load(img_path)
        data.append(img)
        labels.append(idx)

data = np.array(data)
data = data.reshape(data.shape[0], 64, 64, 1)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(os.listdir(preprocessed_dir)), activation='softmax')  # Number of gesture classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('models/gesture_model.h5')
