# Standard library imports
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u

class Cutouts():
    """
    A class for creating and processing cutout images from astronomical data.

    Methods:
    - __init__(header_index, size_arcsec, tolarance_arcsec): 
        Initializes the Cutouts class with header index, cutout size, and tolerance.

    - create_cutouts(image, sky_position): 
        Generates a cutout image centered on the peak flux near a specified sky position.

    - process_cutout_array(sky_position_arr, target_filter_image, secondary_filter_image, avoid_target=False): 
        Processes an array of sky positions to generate cutouts for target and secondary filter images.
    """
    
    def __init__(self, header_index, size_arsec, tolerance_arcsec):
        """
        Parameters:
        - header_index (int): header data index,
        - size_arcsec (float): crop image size.
        - tolerance_arcsec (float): Angular size tolerance (in arcseconds) to find the peak around the center.
        """
        
        self.header_index = header_index
        self.size_arcsec  = size_arsec*u.arcsec
        self.tolerance_arcsec = tolerance_arcsec*u.arcsec
        
    def create_cutouts(self, image, sky_position):
        """
        Create a cutout image around a specified RA/Dec, find the peak flux within a tolerance 
        angular size, recenter a cutout around the peak, and optionally calculate RMS noise.

        Parameters:
        - image: HDUList object containing the image data and header.
        - header_index (int): header data index,
        - sky_position (tuple): RA and Dec of the center to look around, in degrees.
        - size_arcsec (float): crop image size.
        - tolerance_arcsec (float): Angular size tolerance (in arcseconds) to find the peak around the center.

        Returns:
        - cutout_image_target_filt: Cutout of the image centered on the peak flux.
        - peak_position_global: Global pixel coordinates of the peak flux.
        """
        
        # Step 1: Convert RA/Dec to pixel coordinates for the initial centre
        coord = SkyCoord(ra=sky_position[0] * u.deg, dec=sky_position[1] * u.deg, frame='icrs')
        wcs = WCS(image[self.header_index].header, naxis=2)
        initial_center_pix = wcs.world_to_pixel(coord)
        
        # Step 2: Create an initial cutout around the specified RA/Dec position
        initial_cutout = Cutout2D(image[self.header_index].data, initial_center_pix, self.size_arcsec, wcs=wcs)

        # Step 3: Convert the tolerance from arcseconds to pixels
        pixel_scale = np.abs(wcs.pixel_scale_matrix[1, 1]) * u.deg.to(u.arcsec)  # arcseconds per pixel
        tolerance_pixels = self.tolerance_arcsec / pixel_scale

        # Step 4: Create a circular mask within the tolerance radius
        ny, nx = initial_cutout.data.shape
        y, x = np.ogrid[:ny, :nx]
        center_y, center_x = ny // 2, nx // 2
        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = distance_from_center <= tolerance_pixels.value

        # Step 5: Find the peak flux within the masked region
        masked_data = np.where(mask, initial_cutout.data, -np.inf)
        peak_position_cutout = np.unravel_index(np.argmax(masked_data), masked_data.shape)

        # Debug: Verify peak position in the cutout
        print(f"Peak position in cutout: {peak_position_cutout}")

        # Calculate global pixel coordinates of the peak position
        peak_position_global = (
            initial_cutout.origin_original[0] + peak_position_cutout[1],
            initial_cutout.origin_original[1] + peak_position_cutout[0],
        )

        # Step 6: Create new cutouts centred on the peak position
        cutout_image_target_filt = Cutout2D(image[self.header_index].data, peak_position_global, self.size_arcsec, wcs=wcs)
        
        return cutout_image_target_filt, peak_position_global, wcs

    def process_cutout_array(self, sky_position_arr, target_filter_image, secondary_filter_image, avoid_target=False):
        """
        Generate cutouts centered on the local peak within a specified tolerance around RA/Dec positions.

        Parameters:
        - sky_position_arr (list of tuples): List of RA/Dec positions in degrees.
        - target_filter_image (HDUList): Target filter image data.
        - secondary_filter_image (HDUList): Secondary filter image data.
        - avoid_target (bool): If True, return only secondary filter cutouts.

        Returns:
        - cutout_target_image_arr (list): Cutouts of the target filter image.
        - cutout_secondary_image_arr (list): Cutouts of the secondary filter image.
        - err_ind (numpy array): Indices of failed cutouts.
        """
        # Initialize result lists
        cutout_target_image_arr, cutout_secondary_image_arr, error_arr = [], [], []

        # Iterate through each sky position
        for i, sky_position in enumerate(sky_position_arr):
            try:
                # Create cutout for the target filter image
                cutout_image_target_filt, peak_position_global, wcs = self.create_cutouts(
                    image=target_filter_image,
                    sky_position=sky_position)
                # Create cutout for the secondary filter image
                cutout_secondary_filt = Cutout2D(
                    secondary_filter_image[self.header_index].data, peak_position_global, self.size_arcsec, wcs=wcs)

                # Append the resulting cutouts
                cutout_target_image_arr.append(cutout_image_target_filt)
                cutout_secondary_image_arr.append(cutout_secondary_filt)

            except Exception as e:
                # Log the error and index
                cutout_target_image_arr.append(cutout_target_image_arr[-1] if cutout_target_image_arr else None)
                cutout_secondary_image_arr.append(cutout_secondary_image_arr[-1] if cutout_secondary_image_arr else None)
                print(f"Error at index {i}: {e}")
                error_arr.append(i)

        # Convert error indices to numpy array
        error_arr = np.array(error_arr)

        # Return results based on avoid_target flag
        if avoid_target:
            return cutout_secondary_image_arr, error_arr
        return cutout_target_image_arr, cutout_secondary_image_arr, error_arr