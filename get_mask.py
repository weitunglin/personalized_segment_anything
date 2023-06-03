from PIL import Image
import argparse

def create_empty_pixel_mask(image_path, output_path, image_size=None):
    # Open the image
    image = Image.open(image_path)

    # Create a new image with the same size and mode as the original image
    if not image_size:
        mask_image = Image.new("RGB", image.size)
    else:
        image_size = (image_size[0], image_size[1])
        mask_image = Image.new("RGB", image_size)
        image = image.resize(image_size)

    # Load the pixel data of the image
    pixels = image.load()

    # Iterate over each pixel in the image
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            # Get the pixel value
            pixel = pixels[x, y]
            print(pixel)

            # Check if the pixel is transparent
            if pixel[2] == 128:  # Transparent pixel
                # Set the corresponding pixel in the mask image to 1
                mask_image.putpixel((x, y), (0, 0, 128))
            else:
                # Set the corresponding pixel in the mask image to 0
                mask_image.putpixel((x, y), (0, 0, 0))

    # Save the mask image
    mask_image.save(output_path, "PNG")

def main(args):
    create_empty_pixel_mask(args.image_path, args.output_path, args.image_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Program")
    parser.add_argument("--image_path", help="Path to the image file")
    parser.add_argument("--image_size", type=int, nargs="+", help="image size of the image")
    parser.add_argument("--output_path", help="Path to the output mask file")
    args = parser.parse_args()

    main(args)

