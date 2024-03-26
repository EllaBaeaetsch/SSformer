from PIL import Image
import numpy as np
import os

# Path to the directory with the test images
image_directory = 'data/tablesoccer.v2i.coco-mmdetection/train'

# Initialize lists to save the mean values and standard deviations for each image
means = []
stds = []

# Go through all files in the image directory
for file_name in os.listdir(image_directory):
    # Check whether the file is a .jpg file
    if file_name.endswith('.jpg'):
        # Path to the current image file
        file_path = os.path.join(image_directory, file_name)

        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                img_np = np.array(img)

                # Calculate the mean value and the standard deviation for this image
                means.append(img_np.mean(axis=(0, 1)))
                stds.append(img_np.std(axis=(0, 1)))
        except Exception as e:
            print(f"Fehler beim Laden von {file_name}: {e}")

# Calculate the average mean value and the average standard deviation across all images
mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)

mean_rounded = np.round(mean, 3)
std_rounded = np.round(std, 3)

print("Rounded mean value for each channel:", mean_rounded)
print("Rounded standard deviation (Std) for each channel:", std_rounded)
