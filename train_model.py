# /// script
# dependencies = [
#     "tensorflow==2.17.0",
#     "pillow",
#     "scipy",
#     "numpy<2.0.0",
# ]
# ///
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sys

def build_model(num_classes):
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train():
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found or empty.")
        print("Please download a Brain Tumor MRI dataset (e.g., from Kaggle) and extract it to the 'dataset' folder.")
        print("Expected structure:")
        print("  dataset/")
        print("    glioma/")
        print("    meningioma/")
        print("    notumor/")
        print("    pituitary/")
        sys.exit(1)

    batch_size = 32
    epochs = 10
    img_size = (224, 224)

    # Data augmentation and loading
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    class_names = list(train_generator.class_indices.keys())
    print(f"Classes found: {class_names}")

    # Build and compile model
    model = build_model(num_classes=len(class_names))
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Save the model
    model.save('brain_tumor_model.keras')
    print("Model saved to 'brain_tumor_model.keras'")

    # Save class indices
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    print("Class indices saved to 'class_indices.json'")

if __name__ == '__main__':
    train()
