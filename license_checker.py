import piexif
import exifread
import piexif.helper
from PIL import Image
import os
import pathlib

def check_image_license(image_path):
    """
    Check the license of an image by reading its EXIF metadata.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: The license status ("riproducibile" or "diritto d'autore: copia negata")
    """
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
        user_comment = tags.get('EXIF UserComment', None)
        if user_comment:
            return str(user_comment)
        return None

def add_license_to_image(image_path, output_path, license_text):
    """
    Add a license comment to an image's EXIF metadata.
    
    Args:
        image_path (str): Path to the source image
        output_path (str): Path where to save the image with license
        license_text (str): The license text to add
    """
    img = Image.open(image_path)
    exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else {'0th': {}, 'Exif': {}, 'GPS': {}, '1st': {}, 'thumbnail': None}
    
    user_comment = piexif.helper.UserComment.dump(str(license_text))
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
    exif_bytes = piexif.dump(exif_dict)
    
    img.save(output_path, exif=exif_bytes)

def process_dataset_with_licenses(input_dir, output_dir, classes, reproducible_limit=1000):
    """
    Process a dataset of images, adding licenses based on position in the dataset.
    
    Args:
        input_dir (str): Directory containing the input images
        output_dir (str): Directory where to save processed images
        classes (list): List of class names
        reproducible_limit (int): Number of images to mark as reproducible per class
    """
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    
    for class_name in classes:
        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name
        os.makedirs(class_output_dir, exist_ok=True)
        
        images = list(class_input_dir.glob('*'))
        for i, image_path in enumerate(images):
            output_path = class_output_dir / f"{i+1}.jpg"
            if i < reproducible_limit:
                add_license_to_image(str(image_path), str(output_path), "riproducibile")
            else:
                add_license_to_image(str(image_path), str(output_path), "diritto d'autore: copia negata")

def filter_reproducible_images(input_dir, output_dir, classes):
    """
    Filter a dataset to keep only images marked as reproducible.
    
    Args:
        input_dir (str): Directory containing the input images
        output_dir (str): Directory where to save filtered images
        classes (list): List of class names
    """
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    
    for class_name in classes:
        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name
        os.makedirs(class_output_dir, exist_ok=True)
        
        images = list(class_input_dir.glob('*'))
        i = 0
        for image_path in images:
            license_status = check_image_license(str(image_path))
            if license_status == "riproducibile":
                img = Image.open(str(image_path))
                i += 1
                img.save(str(class_output_dir / f"{i}.jpg")) 