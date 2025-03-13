from PIL import Image

def shift_and_blend(image1_path, image2_path, output_path, overlap_width):
    # Load the images
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")
    
    # Calculate the final width of the output image
    final_width = image1.width + image2.width - overlap_width
    final_height = max(image1.height, image2.height)
    
    # Create a blank canvas for the final output
    result = Image.new("RGBA", (final_width, final_height), (0, 0, 0, 0))
    
    # Paste image1 onto the canvas at position (0, 0)
    result.paste(image1, (0, 0))
    
    # Calculate the position to paste image2 for the specified overlap
    image2_position = image1.width - overlap_width
    result.paste(image2, (image2_position, 0))
    
    # Crop the overlapping regions from both images
    image1_overlap = result.crop((image2_position, 0, image1.width, image1.height))
    image2_overlap = result.crop((image2_position, 0, image2_position + overlap_width, image2.height))
    
    # Blend the overlapping areas
    blended_overlap = Image.blend(image1_overlap, image2_overlap, alpha=0.5)
    
    # Paste the blended overlap back onto the result
    result.paste(blended_overlap, (image2_position, 0))
    
    # Save the final blended image
    result.save(output_path, format="PNG")
    print(f"Blended image saved to {output_path}")

# Example usage
overlap_width = 10  # Desired width of the overlapping area
shift_and_blend("image1.png", "image2.png", "output2.png", overlap_width)
