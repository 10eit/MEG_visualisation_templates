import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def crop_brain_and_make_transparent(image):
    """
    Crop the brain figure and make the outside of the brain transparent.
    
    Parameters:
    -----------
    image : pyvista.core.pyvista_ndarray.pyvista_ndarray
        An image array acquired using the `brain.screenshot()` method.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        A matplotlib figure object containing the cropped and transparent image.
    """
    
    # Step 1: Identify non-white pixels in the image, credit to MNE tutorial
    # The image is assumed to have a white background, so we find all pixels that are not white (RGB = [255, 255, 255]).
    nonwhite_pix = (image != 255).any(-1)
    
    # Step 2: Determine the rows and columns that contain non-white pixels
    # This helps in cropping the image to remove the white borders.
    nonwhite_row = nonwhite_pix.any(1)  # Check for any non-white pixels in each row
    nonwhite_col = nonwhite_pix.any(0)  # Check for any non-white pixels in each column
    
    # Step 3: Crop the image to retain only the rows and columns with non-white pixels
    cropped_screenshot = image[nonwhite_row][:, nonwhite_col]
    
    # Step 4: Create an alpha channel for transparency
    # If a pixel is white, it will be made fully transparent (alpha = 0).
    # Otherwise, it will be fully opaque (alpha = 255).
    alpha = np.where(np.all(cropped_screenshot == [255, 255, 255], axis=-1), 0, 255)
    
    # Step 5: Combine the cropped image with the alpha channel
    cropped_screenshot = np.dstack((cropped_screenshot, alpha))
    
    # Step 6: Create a matplotlib figure and axis to display the image
    fig, ax = plt.subplots()
    
    # Step 7: Set the figure background to transparent
    fig.patch.set_alpha(0.0)
    
    # Step 8: Display the cropped and transparent image on the axis
    ax.imshow(cropped_screenshot)
    
    # Step 9: Hide the axis ticks and labels
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    
    # Step 10: Hide the axis spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Step 11: Return the matplotlib figure containing the processed image
    return fig

def align_multiple_brains(brain_list, overlap_ratio=0.2):
    """
    Align multiple brain images horizontally with a specified overlap ratio.
    
    Parameters:
    -----------
    brain_list : list
        A list containing brain images, where each brain image is a matplotlib figure object.
    overlap_ratio : float, default is 0.2
        The ratio of overlap between consecutive images. Should be between 0 and 1.
    
    Returns:
    --------
    new_image : PIL.Image.Image
        A single PIL image containing all the aligned brain images with the specified overlap.
    """
    
    # Initialize an empty list to store the PIL images
    images = list()
    
    # Step 1: Convert each brain figure to a PIL image
    for brain in brain_list:
        # Create a BytesIO buffer to save the figure as a PNG image
        buf = io.BytesIO()
        
        # Save the figure to the buffer with transparent background and tight bounding box
        brain.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        
        # Go to the beginning of the BytesIO buffer
        buf.seek(0)
        
        # Open the image from the buffer using PIL
        image = Image.open(buf)
        
        # Append the PIL image to the list
        images.append(image)
    
    # Step 2: Define parameters for concatenation
    # Calculate the overlap in pixels based on the overlap ratio
    overlap = overlap_ratio
    
    # Get the width and height of the first image (assuming all images have the same dimensions)
    width, height = images[0].size
    
    # Calculate the total width of the concatenated image
    total_width = width * len(images) - int(overlap * width * (len(images) - 1))
    
    # Step 3: Create a new image with a transparent background
    new_image = Image.new('RGBA', (total_width, height))
    
    # Step 4: Paste images with the specified overlap
    for index, image in enumerate(images):
        # Calculate the x-offset for the current image
        x_offset = index * (width - int(overlap * width))
        
        # Paste the current image onto the new image at the calculated offset
        new_image.paste(image, (x_offset, 0), image)
    
    # Step 5: Return the concatenated image
    return new_image
