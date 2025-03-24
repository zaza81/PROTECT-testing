import os
from model_builder import prepare_dataset, build_and_train_model, predict_image

def main():
    # Define paths and parameters
    input_dir = 'intel-image-classification/seg_train/seg_train'
    output_dir = 'dataset'
    classes = ['buildings', 'forest']
    reproducible_limit = 1000
    
    # Prepare the dataset with licenses and filter reproducible images
    print("Preparing dataset...")
    filtered_dir = prepare_dataset(input_dir, output_dir, classes, reproducible_limit)
    
    # Build and train the model
    print("Building and training model...")
    model, history = build_and_train_model(filtered_dir, classes)
    
    # Save the model
    print("Saving model...")
    os.makedirs('saved_model', exist_ok=True)
    model.save('saved_model/my_model.keras')
    
    # Test prediction
    print("Testing prediction...")
    test_image = 'dataset/forest/1.jpg'
    prediction = predict_image(test_image, model, classes)
    print(f"Prediction for {test_image}: {prediction}")

if __name__ == "__main__":
    main() 