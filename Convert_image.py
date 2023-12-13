from PIL import Image

def resize_and_adjust(img_path, output_path, target_size=(28, 28), resolution=(96, 96), new_bit_depth='L'):
    # Load the image
    img = Image.open(img_path)

    # Resize the image to the target size
    img = img.resize(target_size)

    # Change horizontal and vertical resolution
    img.info['dpi'] = resolution

    # Change to 8-bit grayscale
    img = img.convert(new_bit_depth)

    # Save the modified image
    img.save(output_path)

    # Display the modified image
    img.show()

# Example usage:
input_image_path = "img/image.jpg"
output_image_path = "img/new/img_7.jpg"

resize_and_adjust(input_image_path, output_image_path, target_size=(28, 28), resolution=(96, 96), new_bit_depth='L')
