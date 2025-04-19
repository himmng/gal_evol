import numpy as np

class Background_remove():
    """
    Background_remove
    A class for processing astronomical images by removing background noise and filling masked regions with noise.
    Methods:
    --------
    - __init__():
        Initializes the Background_remove class.
    - fill_masked_with_noise_combined(image, source_mask=None, background_mask=None):
    - process_images_with_masks(data_arrays, source_masks, background_masks, good_indices, filt=True):
        - data_arrays (list): List of input data arrays to process.
        - source_masks (list): List of source masks corresponding to each array.
        - background_masks (list): List of background masks corresponding to each array.
        - good_indices (list): Indices of arrays to process.
        - filt (bool, optional): Whether to filter data arrays with a 'data' attribute. Default is True.
        - img_only_noise_list (list): List of images with only noise.
        - img_with_source_list (list): List of images with sources included.
    """
    def __init__(self):
        """_summary_
        pass
        """
        pass
    
    def fill_masked_with_noise_combined(self, image, source_mask=None, background_mask=None):
        """
        This function takes an input image and optionally a source mask and/or a background mask. 
        It fills the masked regions with random noise generated based on the statistical properties 
        (mean and standard deviation) of the unmasked regions of the image.

        - image (numpy.ndarray): The input image to process.
        - source_mask (numpy.ndarray, optional): A boolean mask indicating source regions to be filled with noise. 
            Must have the same shape as the image. Default is None.
        - background_mask (numpy.ndarray, optional): A boolean mask indicating background regions to be filled with noise. 
            Must have the same shape as the image. Default is None.

        - numpy.ndarray (optional): If `background_mask` is provided, an additional image is returned where only 
            the background mask regions are filled with noise.

        Raises:
        - ValueError: If neither `source_mask` nor `background_mask` is provided.
        - ValueError: If the shape of `source_mask` or `background_mask` does not match the shape of the input image.

        Notes:
        - If both `source_mask` and `background_mask` are provided, the masks are combined using a logical OR operation.
        - The noise is generated using a normal distribution with the mean and standard deviation calculated 
            from the unmasked regions of the input image.
        """
        
        if source_mask is None and background_mask is None:
            raise ValueError("At least one of source_mask or background_mask must be provided.")

        if source_mask is not None and image.shape != source_mask.shape:
            raise ValueError("Image and source_mask must have the same shape.")
        if background_mask is not None and image.shape != background_mask.shape:
            raise ValueError("Image and background_mask must have the same shape.")

        # Combine masks if both are provided
        combined_mask = source_mask if background_mask is None else (
            background_mask if source_mask is None else np.logical_or(source_mask, background_mask)
        )

        # Create a masked array
        masked_image = np.ma.masked_array(image, mask=combined_mask)

        # Calculate noise level (standard deviation of unmasked pixels)
        noise_level = np.nanstd(masked_image.compressed())
        noise_loc = np.nanmean(masked_image.compressed())

        # Generate noise for masked pixels
        random_noise = np.random.normal(loc=noise_loc, scale=noise_level, size=image.shape)

        # Create a copy of the original image to modify
        noisy_image = image.copy()
        noisy_image[combined_mask] = random_noise[combined_mask]  # Fill only masked regions

        if background_mask is not None:
            image_with_background_filled = image.copy()
            image_with_background_filled[background_mask] = random_noise[background_mask]
            return noisy_image, image_with_background_filled

        return noisy_image

    def process_images_with_masks(self, data_arrays, source_masks, background_masks, good_indices, filt=True):
        """
        Processes a list of data arrays by applying source and background masks, 
        separating noise and sources.
        This method iterates over the provided data arrays and applies a masking 
        function to generate two outputs: one containing only noise and the other 
        containing sources along with the noise.
        ----------
        data_arrays : list
            A list of input data arrays to process. Each array can be a numpy array 
            or an object with a `.data` attribute.
        source_masks : list
            A list of source masks corresponding to each data array. These masks 
            identify the regions containing sources.
        background_masks : list
            A list of background masks corresponding to each data array. These masks 
            identify the regions containing background noise.
        good_indices : list
            A list of indices specifying which data arrays to process.
        filt : bool, optional
            If True, the method checks if the data array has a `.data` attribute 
            and uses it. Defaults to True.
        -------
        img_only_noise_list : list
            A list of processed images containing only noise, with sources masked out.
        img_with_source_list : list
            A list of processed images containing both noise and sources.
        Raises:
        ------
        Exception
            If an error occurs during processing of a specific index, it is caught 
            and logged, and the method continues processing the remaining indices.
        """
        img_only_noise_list = []
        img_with_source_list = []

        for i in range(len(good_indices)):
            if filt and hasattr(data_arrays[i], 'data'):
                data = data_arrays[i].data
            else:
                data = np.array(data_arrays[i])  # Ensure data is a numpy array
            try:
                
                img_only_noise, img_with_source = self.fill_masked_with_noise_combined(
                    image=data,
                    source_mask=source_masks[i],
                    background_mask=background_masks[i]
                )
                img_with_source_list.append(img_with_source)
                img_only_noise_list.append(img_only_noise)
            except Exception as e:
                print(f"Error processing index {i}: {e}")
                continue

        return img_only_noise_list, img_with_source_list