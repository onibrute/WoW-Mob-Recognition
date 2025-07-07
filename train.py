import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Image size for EfficientNet
image_size = (224, 224)

# Path to dataset
dataset_path = r"C:\Users\Administrator\Desktop\TIA proiect\Mobi"

# Quick configuration
batch_size = 64
epochs = 12
dropout = 0.3
neurons = 128
validation_split = 0.2
learning_rate = 0.0001
frozen_layers = -10

# Load training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=image_size,  # Ensure correct size
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=image_size,  # Ensure correct size
    batch_size=batch_size
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

# Class names
class_names = train_ds.class_names
print("Class names:", class_names)

# Apply data augmentation
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Load EfficientNetB0 as base model, with fine-tuning
base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = True

# Freeze specified layers
for layer in base_model.layers[:frozen_layers]:
    layer.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(dropout),
    layers.Dense(neurons, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the SelectiveCheckpoint callback
class SelectiveCheckpoint(Callback):
    def __init__(self, filepath, validation_data, monitor='val_accuracy', mode='max'):
        super(SelectiveCheckpoint, self).__init__()
        self.filepath = filepath
        self.validation_data = validation_data
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('-inf') if mode == 'max' else float('inf')

        if os.path.exists(self.filepath):
            previous_model = tf.keras.models.load_model(self.filepath)
            _, self.best_score = previous_model.evaluate(self.validation_data, verbose=0)
            print(f"Loaded previous best model {self.monitor}: {self.best_score:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if current_score is not None:
            if (self.mode == 'max' and current_score > self.best_score) or (
                    self.mode == 'min' and current_score < self.best_score):
                print(f"New best {self.monitor}: {current_score:.4f}, saving model.")
                self.best_score = current_score
                self.model.save(self.filepath)
            else:
                print(f"{self.monitor} did not improve: {current_score:.4f} (best: {self.best_score:.4f})")
        else:
            print(f"Warning: Metric '{self.monitor}' not found in logs. Check your model metrics.")

# Path to save the best model
checkpoint_path = r"C:\Users\Administrator\Desktop\TIA proiect\best_model.keras"
custom_checkpoint = SelectiveCheckpoint(filepath=checkpoint_path, validation_data=val_ds,
                                        monitor='val_accuracy',
                                        mode='max')

# Additional callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[custom_checkpoint, early_stopping, reduce_lr]
)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Load and evaluate the best model
best_model = tf.keras.models.load_model(checkpoint_path)
loss, accuracy = best_model.evaluate(val_ds)
print(f"Best model validation accuracy: {accuracy:.4f}")
