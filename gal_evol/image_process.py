import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D

class Center_Cutouts():
    """
    A class for processing astronomical images by centering and cropping them.
    Methods
    -------
    center_cutouts(image, bg_image, wcs, npixels)
        Centers the peak pixel of an image, updates the WCS, and crops the image to a specified size.
    center_cutout_arr(image_arr, bg_image_arr, wcs_arr, npixels)
        Applies the center_cutouts method to an array of images, background images, and WCS objects.
        Centers the peak pixel of an image, updates the WCS, and crops the image to a specified size.
    pad_cutout_to_size(cutout, target_shape)
        Pads a Cutout2D object to a target shape using NaNs and returns a new Cutout2D
        with updated WCS, without re-cropping.
        Parameters
        ----------
        image : numpy.ndarray
            The input 2D image array to be processed.
        bg_image : numpy.ndarray
            The background image used to calculate mean and standard deviation for noise replacement.
        wcs : astropy.wcs.WCS
            The World Coordinate System (WCS) object associated with the image.
        npixels : int
            The size of the square cutout in pixels.
        Returns
        -------
        cutout : astropy.nddata.Cutout2D
            The cropped image and the updated WCS object.
        Applies the center_cutouts method to an array of images, background images, and WCS objects.
        Parameters
        ----------
        image_arr : list of numpy.ndarray
            A list of 2D image arrays to be processed.
        bg_image_arr : list of numpy.ndarray
            A list of background images corresponding to the input images.
        wcs_arr : list of astropy.wcs.WCS
            A list of WCS objects associated with the input images.
        npixels : int
            The size of the square cutout in pixels.
        Returns
        -------
        cutout_arr : list of astropy.nddata.Cutout2D
            A list of cropped images and their updated WCS objects.
    """
    
    def __init__(self):
        pass
    
    def pad_cutout_to_size(self, cutout, target_shape):
        """
        Pads a Cutout2D object to a target shape using NaNs and returns a new Cutout2D
        with updated WCS, without re-cropping.

        Parameters
        ----------
        cutout : Cutout2D
            The input cutout to pad.
        target_shape : tuple of int
            The target shape as (ny, nx).

        Returns
        -------
        Cutout2D
            A new Cutout2D with padded data and corrected WCS.
        """
        current_ny, current_nx = cutout.shape
        target_ny, target_nx = target_shape

        pad_y = max(0, target_ny - current_ny)
        pad_x = max(0, target_nx - current_nx)

        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left

        # Pad data with NaNs
        padded_data = np.pad(
            cutout.data,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=np.nan
        )

        # Update WCS crpix to account for padding
        padded_wcs = cutout.wcs.deepcopy()
        padded_wcs.wcs.crpix[0] += pad_left  # x-direction
        padded_wcs.wcs.crpix[1] += pad_top   # y-direction

        # Build new Cutout2D using manually padded data and adjusted WCS
        # Use position and size corresponding to new full image
        padded_cutout = Cutout2D(
            data=padded_data,
            position=(padded_data.shape[1] // 2, padded_data.shape[0] // 2),
            size=padded_data.shape,
            wcs=padded_wcs,
            mode='trim',  # Use 'trim' since we're not doing a sky cutout anymore
            copy=True
        )

        return padded_cutout

    def center_cutouts(self, target_image, secondary_image, target_bg_image, secondary_bg_image, segment_image, wcs, npixels):
        """
        Centers the peak pixel value of an image, crops it to a specified size, and updates the WCS.

        This function recenters the peak pixel value of the input image to the center of the image,
        crops the image to a specified size using a Cutout2D object, and updates the WCS accordingly.
        Any NaN values in the cropped image are replaced with Gaussian noise based on the background
        image statistics.

        Parameters
        ----------
        target_image : numpy.ndarray
            The input 2D target image array to be processed.
        secondary_image : numpy.ndarray
            The secondary image array to be processed in alignment with the target image.
        target_bg_image : numpy.ndarray
            The background image corresponding to the target image, used to calculate mean and 
            standard deviation for replacing NaN values.
        secondary_bg_image : numpy.ndarray
            The background image corresponding to the secondary image, used to calculate mean and 
            standard deviation for replacing NaN values.
        segment_image : numpy.ndarray
            A segmentation image used to mask the target image for peak detection.
        wcs : astropy.wcs.WCS
            The World Coordinate System (WCS) object associated with the input images.
        npixels : int
            The size of the square cutout in pixels.

        Returns
        -------
        cutout_target_img : astropy.nddata.Cutout2D
            A Cutout2D object containing the cropped target image and the updated WCS.
        cutout_secondary_img : astropy.nddata.Cutout2D
            A Cutout2D object containing the cropped secondary image and the updated WCS.
        cutout_segment : astropy.nddata.Cutout2D
            A Cutout2D object containing the cropped segmentation image and the updated WCS.
        """
        
        # Find the coordinates of the peak pixel value
        img_peak_yy, img_peak_xx = np.unravel_index(np.argmax(np.ma.masked_array(target_image, mask=~segment_image), axis=None), target_image.shape)
        
        # Calculate the center of the image
        centre_xx, centre_yy = np.array(target_image.shape) // 2
        
        # Calculate the shift required to center the peak value
        shift_xx = centre_xx - img_peak_xx
        shift_yy = centre_yy - img_peak_yy
        
        # Shift the image to recenter the peak value
        recentered_target_image = np.roll(target_image, shift=(shift_yy, shift_xx), axis=(0, 1))
        recentered_target_bg_image = np.roll(target_bg_image, shift=(shift_yy, shift_xx), axis=(0, 1))
        recentered_secondary_image = np.roll(secondary_image, shift=(shift_yy, shift_xx), axis=(0, 1))
        recentered_secondary_bg_image = np.roll(secondary_bg_image, shift=(shift_yy, shift_xx), axis=(0, 1))
        recentered_segment_image = np.roll(segment_image, shift=(shift_yy, shift_xx), axis=(0, 1))

        
        # Create an updated WCS object after shifting
        shifted_wcs = wcs.deepcopy()
        shifted_wcs.wcs.crpix[0] += shift_xx  # Update reference pixel X
        shifted_wcs.wcs.crpix[1] += shift_yy  # Update reference pixel Y
        
        # Define the crop center
        cutout_center = (centre_xx, centre_yy)
        
        # Use Cutout2D to crop the image and modify the WCS
        cutout_target_img = Cutout2D(recentered_target_image, position=cutout_center, size=npixels, wcs=shifted_wcs, mode='partial', fill_value=np.nan)
        cutout_target_bg  = Cutout2D(recentered_target_bg_image, position=cutout_center, size=npixels, wcs=shifted_wcs, mode='partial', fill_value=np.nan)
        cutout_secondary_img = Cutout2D(recentered_secondary_image, position=cutout_center, size=npixels, wcs=shifted_wcs, mode='partial', fill_value=np.nan)
        cutout_secondary_bg  = Cutout2D(recentered_secondary_bg_image, position=cutout_center, size=npixels, wcs=shifted_wcs, mode='partial', fill_value=np.nan)
        cutout_segment  = Cutout2D(recentered_segment_image, position=cutout_center, size=npixels, wcs=shifted_wcs, mode='partial', fill_value=0)

        mean_target_bg = np.nanmean(cutout_target_bg.data)
        std_target_bg = np.nanstd(cutout_target_bg.data)
        
        mean_secondary_bg = np.nanmean(cutout_secondary_bg.data)
        std_secondary_bg = np.nanstd(cutout_secondary_bg.data)
        
        # Replace NaN values in the cutout with Gaussian noise from the background
        
        cutout_target_img.data[np.isnan(cutout_target_img.data)] = np.random.normal(mean_target_bg, std_target_bg, np.isnan(cutout_target_img.data).sum())
        cutout_secondary_img.data[np.isnan(cutout_secondary_img.data)] = np.random.normal(mean_secondary_bg, std_secondary_bg, np.isnan(cutout_secondary_img.data).sum())
        

        
        return cutout_target_img, cutout_target_bg, cutout_secondary_img, cutout_secondary_bg, cutout_segment  # Returns the cropped image and the updated WCS

    def center_cutout_arr(self, target_image_arr, secondary_image_arr, target_bg_image_arr, secondary_bg_image_arr, segment_image_arr, wcs_arr, npixels):
        """
        Generate a list of center cutouts for a series of images, along with their updated WCS.

        Parameters
        ----------
        target_image_arr : list of numpy.ndarray
            A list of 2D target image arrays to be processed.
        secondary_image_arr : list of numpy.ndarray
            A list of 2D secondary image arrays to be processed in alignment with the target images.
        target_bg_image_arr : list of numpy.ndarray
            A list of background images corresponding to the target images, used to calculate mean and 
            standard deviation for replacing NaN values.
        secondary_bg_image_arr : list of numpy.ndarray
            A list of background images corresponding to the secondary images, used to calculate mean and 
            standard deviation for replacing NaN values.
        segment_image_arr : list of numpy.ndarray
            A list of segmentation images used to mask the target images for peak detection.
        wcs_arr : list of astropy.wcs.WCS
            A list of WCS (World Coordinate System) objects corresponding to the input images.
        npixels : int
            The size (in pixels) of the square cutout to extract from the center of each image.

        Returns
        -------
        cutout_target_img_arr : list of astropy.nddata.Cutout2D
            A list of Cutout2D objects containing the cropped target images and their updated WCS.
        cutout_secondary_img_arr : list of astropy.nddata.Cutout2D
            A list of Cutout2D objects containing the cropped secondary images and their updated WCS.
        cutout_segment_arr : list of astropy.nddata.Cutout2D
            A list of Cutout2D objects containing the cropped segmentation images and their updated WCS.
        """
        cutout_target_img_arr, cutout_target_bg_arr, cutout_secondary_img_arr, cutout_secondary_bg_arr, cutout_segment_arr = [], [], [], [], []
        for i in range(len(target_image_arr)):
            cutout = self.center_cutouts(target_image_arr[i], secondary_image_arr[i], target_bg_image_arr[i], secondary_bg_image_arr[i], segment_image_arr[i], wcs_arr[i].wcs, npixels)
            cutout_target_img_arr.append(cutout[0])
            cutout_target_bg_arr.append(cutout[1])
            cutout_secondary_img_arr.append(cutout[2])
            cutout_secondary_bg_arr.append(cutout[3])
            cutout_segment_arr.append(cutout[4])
        return cutout_target_img_arr, cutout_target_bg_arr, cutout_secondary_img_arr, cutout_secondary_bg_arr, cutout_segment_arr  # Returns the cropped image and the updated WCS
    
class Save_Data():
    """
    A class for saving processed astronomical image data and metadata to FITS files.

    Methods
    -------
    save_fits(sample_type, target_filter_name, secondary_filter_name, savepath)
        Saves the processed image data, segmentation masks, and metadata to a FITS file.
    """
    
    def __init__(self,  target_image_arr    
                        ,secondary_image_arr   
                        ,target_segment_arr    
                        ,target_bg_arr         
                        ,secondary_bg_arr      
                        ,target_original_arr   
                        ,secondary_original_arr
                        ,sample_data):
        """
        Initializes the Save_Data class with the provided image arrays and sample data.

        Parameters
        ----------
        target_image_arr : list of numpy.ndarray
            A list of 2D arrays representing the processed target images.
        secondary_image_arr : list of numpy.ndarray
            A list of 2D arrays representing the processed secondary images.
        target_segment_arr : list of numpy.ndarray
            A list of 2D arrays representing the segmentation masks for the target images.
        target_bg_arr : list of numpy.ndarray
            A list of 2D arrays representing the background images for the target images.
        secondary_bg_arr : list of numpy.ndarray
            A list of 2D arrays representing the background images for the secondary images.
        target_original_arr : list of numpy.ndarray
            A list of 2D arrays representing the original target images.
        secondary_original_arr : list of numpy.ndarray
            A list of 2D arrays representing the original secondary images.
        sample_data : numpy structured array
            A structured array containing metadata or additional data related to the images.
        """
        self.target_wcs_arr         = np.array([target_image_arr      [i].wcs  for i in range(len(target_image_arr      ))])
        self.target_image_arr       = np.array([target_image_arr      [i].data for i in range(len(target_image_arr      ))])
        self.secondary_image_arr    = np.array([secondary_image_arr   [i].data for i in range(len(secondary_image_arr   ))])
        self.target_segment_arr     = np.array([target_segment_arr    [i].data for i in range(len(target_segment_arr    ))])
        self.target_bg_arr          = np.array([target_bg_arr         [i].data for i in range(len(target_bg_arr         ))])
        self.secondary_bg_arr       = np.array([secondary_bg_arr      [i].data for i in range(len(secondary_bg_arr      ))])
        self.target_original_arr    = np.array([target_original_arr   [i].data for i in range(len(target_original_arr   ))])
        self.secondary_original_arr = np.array([secondary_original_arr[i].data for i in range(len(secondary_original_arr))])
        self.sample_data            = sample_data
        
    def save_fits(self, sample_type, target_filter_name, secondary_filter_name, savepath, skip_target=False):
        """
        Saves the processed image data, segmentation masks, and metadata to a FITS file.

        This method creates a FITS file containing the original images, segmentation masks, 
        cutouts, background images, WCS information, and structured data fields. Each component 
        is stored in a separate HDU (Header Data Unit) with appropriate headers.

        Parameters
        ----------
        sample_type : str
            The type of sample being processed (e.g., 'galaxy', 'star').
        target_filter_name : str
            The name of the filter used for the target images (e.g., 'F606W').
        secondary_filter_name : str
            The name of the filter used for the secondary images (e.g., 'F814W').
        savepath : str
            The directory path where the FITS file will be saved.

        Returns
        -------
        None
        """
        
        if skip_target==False:

            # Primary HDU: target filter original image
            primary_hdu = fits.PrimaryHDU(self.target_original_arr)
            primary_hdu.header['EXTNAME'] = f'{target_filter_name}:original-image'
            
            # secondary filter original image
            secondary_hdu = fits.ImageHDU(self.secondary_original_arr)
            secondary_hdu.header['EXTNAME'] = f'{secondary_filter_name}:original-image'

            # segment HDU
            mask_hdu = fits.ImageHDU(np.array(self.target_segment_arr).astype(np.uint8))
            mask_hdu.header['EXTNAME'] = f'segmentation-masks'

            # Create ImageHDUs with the stacked arrays (without WCS headers)
            filter0_hdu = fits.ImageHDU(self.target_image_arr)
            filter0_hdu.header['EXTNAME'] = f'{target_filter_name}-cutouts'

            filter1_hdu = fits.ImageHDU(self.secondary_image_arr)
            filter1_hdu.header['EXTNAME'] = f'{secondary_filter_name}-cutouts'

            filter0bg_hdu = fits.ImageHDU(self.target_bg_arr)
            filter0bg_hdu.header['EXTNAME'] = f'{target_filter_name}-background'

            filter1bg_hdu = fits.ImageHDU(self.secondary_bg_arr)
            filter1bg_hdu.header['EXTNAME'] = f'{secondary_filter_name}-background'
            
            # Create an HDU for each WCS object and add it to the list of HDUs
            wcs_hdus = [] # additional wcs 
            for i in range(len(self.target_wcs_arr)):  # Dynamically based on the length of the data
                wcs_hdu = fits.ImageHDU(np.zeros((1, 1)))  # Placeholder data for WCS HDUs
                wcs_hdu.header['EXTNAME'] = f'WCS'
                wcs_hdu.header.update(self.target_wcs_arr[i].to_header())  # Add WCS header
                wcs_hdus.append(wcs_hdu)
                
            structured_data_hdus = []
            for field_name in self.sample_data.dtype.names:
                field_data = self.sample_data[field_name]
                field_hdu = fits.ImageHDU(field_data)
                field_hdu.header['EXTNAME'] = field_name  # Use field name as EXTNAME
                structured_data_hdus.append(field_hdu)

            # Create HDU list and save to a FITS file
            hdul = fits.HDUList([
                primary_hdu, secondary_hdu, mask_hdu, filter0_hdu, filter1_hdu, 
                filter0bg_hdu, filter1bg_hdu,
            ] + wcs_hdus + structured_data_hdus)  # Append WCS HDUs to the list

            hdul.writeto(savepath+f'{sample_type}_final_filters_{target_filter_name}_{secondary_filter_name}.fits', overwrite=True)
            
            print(f"FITS file {sample_type}_final_filters_{target_filter_name}_{secondary_filter_name}.fits created successfully.")
        
    
        else:
            
            # secondary filter original image
            secondary_hdu = fits.PrimaryHDU(self.secondary_original_arr)
            secondary_hdu.header['EXTNAME'] = f'{secondary_filter_name}:original-image'
            
            filter1_hdu = fits.ImageHDU(self.secondary_image_arr)
            filter1_hdu.header['EXTNAME'] = f'{secondary_filter_name}-cutouts'

            filter1bg_hdu = fits.ImageHDU(self.secondary_bg_arr)
            filter1bg_hdu.header['EXTNAME'] = f'{secondary_filter_name}-background'
            
            # Create HDU list and save to a FITS file
            hdul = fits.HDUList([
                secondary_hdu, filter1_hdu, filter1bg_hdu,
            ])  # Append WCS HDUs to the list

            hdul.writeto(savepath+f'{sample_type}_final_filters_{secondary_filter_name}.fits', overwrite=True)

            print(f"FITS file {sample_type}_final_filters_{secondary_filter_name}.fits created successfully.")
        
        
