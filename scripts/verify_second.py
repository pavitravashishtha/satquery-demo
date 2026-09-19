import glob
from PIL import Image

images = glob.glob('/home/pavitra/satquery/data/SECOND/im1/*.png')
print(f"Total images in im1: {len(images)}")

img2_images = glob.glob('/home/pavitra/satquery/data/SECOND/im2/*.png')
print(f"Total images in im2: {len(img2_images)}")

if images:
    Image.open(images[0]).verify()
    print(f"Sample image verified successfully: {images[0]}")
