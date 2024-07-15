from PIL import Image
import os

def compress_image(input_path, output_path, downscale_factor, compression_method="bicubic"):
    """
    Compresses an image with a given downscale factor and compression method, preserving the
    original filename and adding a suffix indicating the downscale factor.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Directory for the compressed output image.
        downscale_factor (int): Factor by which to downscale the image (e.g., 4 for 1/4 size).
        compression_method (str, optional): Method for resizing (default: "bicubic").

    Raises:
        ValueError: If an unsupported compression method is chosen.
    """

    try:
        img = Image.open(input_path)
        original_width, original_height = img.size
        new_width = int(original_width / downscale_factor)
        new_height = int(original_height / downscale_factor)

        resized_img = img.resize((new_width, new_height), getattr(Image, compression_method.upper()))

        # Construct the output filename with the downscale factor
        filename, ext = os.path.splitext(os.path.basename(input_path))
        output_filename = f"{filename}_downsized_{downscale_factor}x{ext}"
        output_path = os.path.join(output_path, output_filename)

        # Ensure saving as BMP
        output_path = output_path if output_path.endswith(".bmp") else output_path + ".bmp"
        resized_img.save(output_path, "BMP")

        print(f"Image compressed and saved as: {output_path}")
    except AttributeError:
        raise ValueError(f"Invalid compression method: {compression_method}")

#                       !!!HERE YOU HAVE DO UPDATE 2 PATHS!!!
input_path = "C:/Users/maxmi/Desktop/image_downsample/images/zebra.jpg"
output_path = "C:/Users/maxmi/Desktop/image_downsample/images/"
downscale_factor = 2

compress_image(input_path, output_path, downscale_factor)  # Default to bicubic compression
