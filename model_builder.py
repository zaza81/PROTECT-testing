import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
from license_checker import process_dataset_with_licenses, filter_reproducible_images

def build_and_train_model(data_dir, classes, batch_size=32, img_height=228, img_width=228, epochs=15):
    """
    Build and train a CNN model for image classification.
    
    Args:
        data_dir (str): Directory containing the training data
        classes (list): List of class names
        batch_size (int): Batch size for training
        img_height (int): Height of input images
        img_width (int): Width of input images
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (trained model, training history)
    """
    # Create training and validation datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Configure dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build the model
    num_classes = len(classes)
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return model, history

def predict_image(filename, model, class_names):
    """
    Make a prediction for a single image.
    
    Args:
        filename (str): Path to the image file
        model: Trained TensorFlow model
        class_names (list): List of class names
        
    Returns:
        str: Predicted class name
    """
    img_ = image.load_img(filename, target_size=(228, 228))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, 0)
    predictions = model.predict(img_processed)
    score = tf.nn.softmax(predictions[0])
    index = np.argmax(score)
    return class_names[index]

def prepare_dataset(input_dir, output_dir, classes, reproducible_limit=1000):
    """
    Prepare the dataset by adding licenses and filtering reproducible images.
    
    Args:
        input_dir (str): Directory containing the original images
        output_dir (str): Directory where to save processed images
        classes (list): List of class names
        reproducible_limit (int): Number of images to mark as reproducible per class
    """
    # First process all images with licenses
    process_dataset_with_licenses(input_dir, output_dir, classes, reproducible_limit)
    
    # Then filter to keep only reproducible images
    filtered_dir = output_dir + "_filtered"
    filter_reproducible_images(output_dir, filtered_dir, classes)
    
    return filtered_dir 