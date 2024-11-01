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

def convert_all(brain_list,overlap_ratio=0.2):
    """
    Convert a list of brain screenshots into a single aligned and transparent image.
    
    Parameters:
    -----------
    brain_list : list
        A list of brain screenshots, where each screenshot is a numpy array.
    overlap_ratio : float, default is 0.2
        The ratio of overlap between consecutive images. Should be between 0 and 1.
        
    
    Returns:
    --------
    aligned : PIL.Image.Image
        A single PIL image containing all the aligned and transparent brain screenshots.
    """
    
    # Step 1: Initialize an empty list to store the processed images
    images = []
    
    # Step 2: Iterate over each screenshot in the brain_list
    for screenshot in brain_list:
        # Step 3: Crop the brain figure and make the outside of the brain transparent
        # This is done using the previously defined function crop_brain_and_make_transparent
        image = crop_brain_and_make_transparent(screenshot)
        
        # Step 4: Append the processed image to the list of images
        images.append(image)
    
    # Step 5: Align multiple brain images horizontally with a specified overlap ratio
    # This is done using the previously defined function align_multiple_brains
    aligned = align_multiple_brains(images, overlap_ratio=overlap_ratio)
    
    # Step 6: Return the aligned and transparent image
    return aligned

import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from io import BytesIO
from PIL import Image

def concatenate_hemis(lh, rh):
    """
    Concatenate two hemispheres (left and right) vertically into a single image.
    
    Parameters:
    -----------
    lh : PIL.Image.Image
        The image of the left hemisphere.
    rh : PIL.Image.Image
        The image of the right hemisphere.
    
    Returns:
    --------
    result_image : PIL.Image.Image
        A single PIL image containing both hemispheres concatenated vertically.
    """
    
    # Step 1: Calculate the aspect ratio of the left hemisphere image
    ratio = lh.size[0] / lh.size[1]
    
    # Step 2: Create a figure with two subplots arranged vertically
    # The figure size is adjusted based on the aspect ratio of the left hemisphere
    fig = plt.figure(figsize=(math.floor(2 * ratio), 4))
    
    # Step 3: Create an ImageGrid to arrange the subplots
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 1),  # creates 2x1 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     share_all=True,
                     )
    
    # Step 4: Turn off interactive mode to avoid displaying the figure
    plt.ioff()
    
    # Step 5: Display the left hemisphere image in the first subplot
    grid[0].imshow(lh)
    grid[0].axis("off")  # Hide the axis
    
    # Step 6: Display the right hemisphere image in the second subplot
    grid[1].imshow(rh)
    grid[1].axis("off")  # Hide the axis
    
    # Step 7: Adjust the layout of the figure
    fig.subplots_adjust(left=0, right=1.0, bottom=0.01, top=0.9, wspace=0.1, hspace=0.5)
    
    # Step 8: Load the figure into a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    
    # Step 9: Close the figure to free memory
    plt.close(fig)
    
    # Step 10: Move to the beginning of the BytesIO buffer
    buf.seek(0)
    
    # Step 11: Create a PIL Image from the BytesIO object
    result_image = Image.open(buf)
    
    # Step 12: Return the concatenated image
    return result_image

import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mne

def add_colorbar(clim, aligned, label, filename, caption):
    """
    Add a colorbar to an aligned brain image and save the result as a PNG file.
    
    Parameters:
    -----------
    clim : dict
        Color limits for the colorbar.
    aligned : PIL.Image.Image
        The aligned brain image to which the colorbar will be added.
    label : str
        Label for the colorbar.
    filename : str
        Filename for saving figure.
    caption : str
        Caption for the figure.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the aligned image and colorbar.
    """
    
    # Step 1: Define the colormap
    cmap = 'inferno'
    
    # Step 2: Calculate the aspect ratio of the aligned image
    ratio = aligned.size[0] / aligned.size[1]
    
    # Step 3: Create a figure with a single subplot
    # The figure size is adjusted based on the aspect ratio of the aligned image
    fig, ax = plt.subplots(figsize=(math.floor(2 * ratio), 4))
    
    # Step 4: Turn off interactive mode to avoid displaying the figure
    plt.ioff()
    
    # Step 5: Display the aligned image in the subplot
    ax.imshow(aligned)
    ax.axis("off")  # Hide the axis
    
    # Step 6: Create a divider and append an axes for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="0.5%", pad=0.15)
    
    # Step 7: Plot the colorbar using mne.viz.plot_brain_colorbar
    cbar = mne.viz.plot_brain_colorbar(cax, clim, cmap, label=label)
    
    # Step 8: Adjust the layout of the figure
    fig.subplots_adjust(left=0, right=1.0, bottom=0.01, top=0.9, wspace=0.1, hspace=0.5)
    
    # Step 9: Add a caption to the figure
    fig.suptitle(caption, fontsize=18, fontweight='bold', y=0.8)
    
    # Step 10: Save the figure as a PNG file with transparency
    fig.savefig(f"{filename}.png", transparent=True)
    
    # Step 11: Return the figure
    return fig

def pipeline(stc_path, stc_subject, fs_path, subject_dir, t_thresh, label, filename, caption):
    """
    Execute a pipeline to process source estimate (STC) data, generate brain screenshots,
    align and concatenate hemispheres, and add a colorbar to the final image.
    
    Parameters:
    -----------
    stc_path : str
        Path to the source estimate (STC) file.
    stc_subject : str
        Subject ID for the STC data.
    fs_path : str
        Path to the fsaverage source space file.
    subject_dir : str
        Path to the subjects directory.
    t_thresh : float
        Threshold value for the color limits.
    label : str
        Label for the colorbar.
    filename : str
        Filename for saving figure.
    caption : str
        Caption for the figure.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The final figure containing the concatenated hemispheres and colorbar.
    """
    
    # Step 1: Initialize an empty list to store the aligned hemispheres
    hemis = []
    
    # Step 2: Process both hemispheres ('lh' for left hemisphere and 'rh' for right hemisphere)
    for hemi in ['lh', 'rh']:
        # Step 3: Generate brain screenshots for the current hemisphere
        brain_list, clim = stc_to_screenshots(stc_path=stc_path, stc_subject=stc_subject, fs_path=fs_path, subject_dir=subject_dir, hemi=hemi, t_thresh=t_thresh)
        
        # Step 4: Convert all brain screenshots into a single aligned image
        aligned = convert_all(brain_list)
        
        # Step 5: Append the aligned image to the list of hemispheres
        hemis.append(aligned)
    
    # Step 6: Extract the aligned images for the left and right hemispheres
    lh, rh = hemis[0], hemis[1]
    
    # Step 7: Concatenate the left and right hemispheres vertically
    concated_hemis = concatenate_hemis(lh, rh)
    
    # Step 8: Add a colorbar to the concatenated hemispheres and save the result as a PNG file
    fig = add_colorbar(clim, concated_hemis, label, filename, caption)
    
    # Step 9: Return the final figure
    return fig
