import os
from PIL import Image

def resize_image(image_path, output_path, size=(64, 64)):
    """Resize the image to the given size and save it."""
    try:
        with Image.open(image_path) as img:
            img_resized = img.resize(size)
            img_resized.save(output_path)
            print(f"Resized and saved: {output_path}")
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")

def resize_images_in_directory(base_dir, size=(64, 64)):
    """Traverse through all directories and resize images."""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file is an image (based on extension)
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                # Create a new output path for the resized image
                output_path = os.path.join(root, file)  # overwrite original, or modify to create a new directory
                resize_image(image_path, output_path, size)

if __name__ == "__main__":
    # Set the base directory to the folder where your datasets are
    base_directory = './dataset5'  # Change this path as necessary
    resize_images_in_directory(base_directory)

