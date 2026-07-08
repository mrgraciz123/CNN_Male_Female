import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models

def parse_args():
    parser = argparse.ArgumentParser(description="Train NEXUS-CNN Gender Classifier")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="Path to the dataset directory containing 'female' and 'male' folders"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=150,
        help="Target size to resize images (default: 150)"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation (default: 0.2)"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="gender_classifier.keras",
        help="File path to save the trained model (default: gender_classifier.keras)"
    )
    return parser.parse_args()

def create_model(img_width, img_height):
    # CNN Model Architecture as defined in the project
    model = models.Sequential()
    
    # 1. Conv2D block 1 (32 filters, 3x3 kernel, ReLU, input 150x150x3)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 2. Conv2D block 2 (64 filters, 3x3 kernel, ReLU)
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 3. Conv2D block 3 (128 filters, 3x3 kernel, ReLU)
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten & Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile with Adam optimizer and binary crossentropy loss
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    args = parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    print(f"Loading dataset from: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return

    # Check subfolders to ensure classes are present
    subfolders = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
    print(f"Found classes/folders: {subfolders}")
    if not ('female' in subfolders or 'male' in subfolders):
        print("Warning: Expected folders 'female' and/or 'male' not found in dataset directory.")

    # Load training and validation datasets
    print("Loading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=args.validation_split,
        subset="training",
        seed=123,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="binary"
    )

    print("Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=args.validation_split,
        subset="validation",
        seed=123,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="binary"
    )

    # Class names mapping check
    class_names = train_ds.class_names
    print(f"Dataset classes mapped: {class_names}")

    # Standard normalization mapping: scale inputs to [0, 1] range
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Configure dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Creating the CNN model
    print("Initializing model...")
    model = create_model(args.img_size, args.img_size)
    model.summary()

    # Train the model
    print(f"Starting model training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds
    )

    # Save the model
    output_path = args.output_model
    print(f"Saving model to {output_path}...")
    model.save(output_path)
    print("Model training and saving completed successfully!")

if __name__ == "__main__":
    main()
