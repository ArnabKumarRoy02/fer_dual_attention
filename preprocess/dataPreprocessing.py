import os
import argparse
from PIL import Image

def resize_image(root_folder, target_size):
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                try:
                    img = Image.open(file_path)
                    img = img.resize(target_size, Image.ANTIALIAS)
                    img.save(file_path)
                    print(f"Resized {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        else:
            try:
                img = Image.open(item_path)
                img = img.resize(target_size)
                img.save(item_path)
                print(f"Resized {item_path}")
            except Exception as e:
                print(f"Error processing {item_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--width', type=int, default=112, help='Width of the image')
    parser.add_argument('--height', type=int, default=112, help='Height of the image')
    args = parser.parse_args()

    resize_image(args.data_path, target_size=(args.width, args.height))

if __name__ == '__main__':
    main()
