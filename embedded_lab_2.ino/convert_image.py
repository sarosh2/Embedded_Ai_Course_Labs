import cv2
import numpy as np
import os

def preprocess_image(image_path: str, target_size=(28, 28)) -> np.ndarray:
    """Reads and preprocesses the image by resizing and converting to grayscale."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def save_to_header(images: list, output_file: str):
    """Flattens the image array in row-major order and writes it as a C header file in int32 array format."""
    with open(output_file, 'w') as f:
        f.write('#ifndef IMAGE_DATA_H\n')
        f.write('#define IMAGE_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        
        num_images = len(images)
        f.write(f'const int num_images = {num_images};\n')
        f.write('const int image_data[][28*28] = {\n')

        for image in images:
            f.write('    {')
            for row in image:
                for value in row:
                    f.write(f'{value}, ')
            f.write('},\n')

        f.write('};\n\n')
        f.write('#endif // IMAGE_DATA_H\n')

if __name__ == "__main__":
    # Define the paths for the images and the output file name with .h extension (e.g., filename.h)
    image_dir = "test_inputs"
    output_file = "input.h"

    # List of image paths
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

    # Preprocess images
    processed_images = [preprocess_image(image_path) for image_path in image_paths]

    # Save to header file
    save_to_header(processed_images, output_file)
