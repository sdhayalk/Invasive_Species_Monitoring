import os
import cv2

def resize_all_images(directory, d1, d2):
    for current_dir in os.walk(directory):
        for current_file in current_dir[2]:
            current_path_with_file = directory + "/" + current_file

            img = cv2.imread(current_path_with_file)
            resized_img = cv2.resize(img, (d1, d2))
            cv2.imwrite(current_path_with_file, resized_img)

def rotate_image(directory_in, directory_out, degree):
    for current_dir in os.walk(directory_in):
        for current_file in current_dir[2]:
            current_path_with_file = directory_in + "/" + current_file

            img = cv2.imread(current_path_with_file)
            num_rows, num_cols = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), degree, 1)
            img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
            cv2.imwrite(directory_out + "/" + current_file, img)

def flip_image_vertically(directory_in, directory_out):
    for current_dir in os.walk(directory_in):
        for current_file in current_dir[2]:
            current_path_with_file = directory_in + "/" + current_file

            img = cv2.imread(current_path_with_file)
            img = cv2.flip(img, 1)
            cv2.imwrite(directory_out + "/" + current_file, img)

# resize_all_images('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train', 224, 224)
# resize_all_images('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test', 224, 224)

if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_90'):
    os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_90')
rotate_image('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_90', 90)

# if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_90'):
#     os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_90')
# rotate_image('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_90', 90)


if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_180'):
    os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_180')
rotate_image('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_180', 180)

# if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_180'):
#     os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_180')
# rotate_image('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_180', 180)


if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_270'):
    os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_270')
rotate_image('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_270', 270)

# if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_270'):
#     os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_270')
# rotate_image('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_270', 270)

if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_flip'):
    os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_flip')
flip_image_vertically('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/train_flip')

# if not os.path.exists('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_flip'):
#     os.makedirs('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_flip')
# flip_image_vertically('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test', 'G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test_flip')