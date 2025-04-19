import numpy as np
import warnings
from scipy.ndimage import convolve
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
from photutils.utils import NoDetectionsWarning
from photutils.background import Background2D, SExtractorBackground, MedianBackground
from photutils.segmentation import SegmentationImage

class Segmentation():
    """
    A class for performing image segmentation and deblending.

    This class provides methods to process astronomical images, detect sources, 
    subtract background, convolve data, and generate segmentation maps with 
    deblending capabilities. It also includes utilities for handling arrays of 
    images and applying masks based on segmentation results.
    """
    
    def __init__(self, use_median_background, box_size, filt_size, npixels_segmentation, npixels_deblend, sigma_clip, nlevels, contrast, gaussian_fwhm=2.5, gaussian_size=3):
        """
        Initialize the Segmentation class.

        Args:
            use_median_background (bool): If True, use MedianBackground for background estimation; 
                           otherwise, use SExtractorBackground.
            box_size (int): Size of the box for background estimation.
            filt_size (int): Size of the filter for background smoothing.
            npixels_segmentation (int): Minimum number of connected pixels required for source detection.
            npixels_deblend (int): Minimum number of connected pixels required for source deblending.
            sigma_clip (float): Sigma value for sigma-clipped statistics.
            nlevels (int): Number of multi-thresholding levels for deblending.
            contrast (float): Minimum contrast ratio for deblending sources.
            gaussian_fwhm (float, optional): Full-width at half-maximum of the Gaussian kernel. Defaults to 2.5.
            gaussian_size (int, optional): Size of the Gaussian kernel. Defaults to 3.
        """

        if use_median_background:
            self.bkg_estimator = MedianBackground()
        else:
            self.bkg_estimator = SExtractorBackground()
            
        self.box_size = box_size
        self.filt_size = filt_size
        self.npixels_deblend = npixels_deblend
        self.npixels_segmentation = npixels_segmentation
        self.sigma_clip = sigma_clip
        self.nlevels = nlevels
        self.contrast = contrast
        self.kernel = make_2dgaussian_kernel(fwhm=gaussian_fwhm, size=gaussian_size)
        
    def get_segmentation_with_deblend(self, image,):
        """"
        Perform image segmentation with deblending.
        This method processes an input image to detect sources, subtracts the background,
        convolves the data, and performs segmentation and deblending to identify individual sources.
            image (astropy.nddata.CCDData): The input image data to be segmented.
            tuple: A tuple containing the following:
                - segment_deblended (numpy.ndarray): The deblended segmentation map.
                - segment_map (numpy.ndarray): The initial segmentation map.
                - data_bg_sub (numpy.ndarray): The background-subtracted image data.
                - convolved_data (numpy.ndarray): The convolved image data.
        Raises:
            Exception: If any error occurs during the segmentation process, it is caught and printed.
        """
        
        try:
            bkg = Background2D(image.data, box_size=(self.box_size, self.box_size),
                            filter_size=(self.filt_size, self.filt_size), bkg_estimator=self.bkg_estimator)
            data_bg_sub = image.data - bkg.background
            convolved_data = convolve(data_bg_sub, self.kernel)
            _, _, std = sigma_clipped_stats(data_bg_sub, sigma=self.sigma_clip)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NoDetectionsWarning)
                segment_map = detect_sources(convolved_data, threshold=std, npixels=self.npixels_segmentation)
                if segment_map is None:
                    print(f"No sources detected. Skipping this cutout.")
                    segment_map = SegmentationImage(np.ones(convolved_data.shape, dtype=int))
                    segment_deblended = segment_map
                else:
                    segment_deblended = deblend_sources(convolved_data, segment_map,
                                                        npixels=self.npixels_deblend, nlevels=self.nlevels, contrast=self.contrast,
                                                        progress_bar=False)
            
        except Exception as e:
            print(f"Error: {e}")

        return segment_deblended, segment_map, data_bg_sub, convolved_data
    
    def get_deblend_segmentation(self, convolved_image, segment_image):
        """
        Deblend sources with a segmented image.

        Args:
            convolved_image (numpy.ndarray): The convolved image data.
            segment_image (numpy.ndarray): The initial segmentation map.

        Returns:
            numpy.ndarray: The deblended segmentation map.
        """
        
        try:
            segment_deblended = deblend_sources(convolved_image, segment_image,
                                                npixels=self.npixels_deblend, nlevels=self.nlevels, contrast=self.contrast,
                                                progress_bar=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NoDetectionsWarning)
                if segment_image is None:
                        print(f"No sources detected. Skipping this cutout.")
                        segment_image = SegmentationImage(np.ones(convolved_image.shape, dtype=int))
            
        except Exception as e:
            print(f"Error: {e}")
        
        return segment_deblended
    
    def get_segmentations_arr(self, image_arr):
        """
        Processes an array of images to generate segmentation maps, background-subtracted data, 
        and convolved data, while handling errors for individual images.
            image_arr (numpy.ndarray): An array of image cutouts to process.
            tuple: A tuple containing the following elements:
                - deblend_segment_map_arr (list): List of deblended segmentation maps for each image.
                - segment_map_arr (list): List of segmentation maps for each image.
                - data_bg_sub_arr (list): List of background-subtracted data arrays for each image.
                - data_bg_sub_arr_convolved (list): List of convolved background-subtracted data arrays for each image.
                - err_ind (numpy.ndarray): Array of indices where errors occurred during processing.
        """
        
        results = {
            "deblend_segment_map_arr": [],
            "segment_map_arr": [],
            "data_bg_sub_arr": [],
            "data_bg_sub_arr_convolved": [],
            "err_ind": []
        }
        
        for i, cutout in enumerate(image_arr):
            try:
                segment_deblended, segment_map, data_bg_sub, convolved_data = self.get_segmentation_with_deblend(cutout)
                results["data_bg_sub_arr"].append(data_bg_sub)
                results["data_bg_sub_arr_convolved"].append(convolved_data)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=NoDetectionsWarning)
                    if segment_map is None:
                        print(f"No sources detected. Skipping this cutout.")
                        segment_map = SegmentationImage(np.ones(convolved_data.shape, dtype=int))
                        segment_deblended = segment_map 
                    else:
                        results["deblend_segment_map_arr"].append(segment_deblended)
                        results["segment_map_arr"].append(segment_map)
                
            except Exception as e:
                print(f"Error at index {i}: {e}")
                results["err_ind"].append(i)

        results["err_ind"] = np.unique(results["err_ind"])
        
        return (results["deblend_segment_map_arr"], results["segment_map_arr"], results["data_bg_sub_arr"], results["data_bg_sub_arr_convolved"], results["err_ind"])
    
    
    def apply_closest_mask(self, data, segment):
        """
        Apply the closest mask to the data based on the segmentation.

        Args:
            data (numpy.ndarray): The input data array (e.g., image data).
            segment (photutils.segmentation.SegmentationImage): The segmentation object containing labeled regions.

        Raises:
            ValueError: If the 'segment' parameter is not a valid SegmentationImage object.
            ValueError: If the segmentation object does not contain at least one label.

        Returns:
            tuple: A tuple containing the following:
            - masked_image (numpy.ndarray): The input data masked with the closest segment.
            - closest_mask (numpy.ndarray): A boolean mask for the closest segment.
            - merged_other_masks (numpy.ndarray): A boolean mask for all other segments merged together.
        """
        centre_y, centre_x = np.array(data.shape) // 2

        if not hasattr(segment, 'nlabels') or not hasattr(segment, 'data'):
            raise ValueError("The 'segment' parameter must be a valid SegmentationImage object.")

        nlabels = segment.nlabels
        if nlabels < 1:
            raise ValueError("Segmentation object must have at least one label.")

        if nlabels == 1:
            closest_mask = segment.data == 1
            merged_other_masks = np.zeros(data.shape, dtype=bool)
            masked_image = np.where(closest_mask, data, np.nan)
            return masked_image, closest_mask, merged_other_masks

        segment_masks = [segment.data == label for label in range(1, nlabels + 1)]
        distances = [
            np.min(np.sqrt((np.where(mask)[1] - centre_x) ** 2 + (np.where(mask)[0] - centre_y) ** 2))
            for mask in segment_masks
        ]
        closest_mask_ind = int(np.argmin(distances))
        closest_mask = segment_masks[closest_mask_ind]
        merged_other_masks = np.logical_or.reduce([mask for idx, mask in enumerate(segment_masks) if idx != closest_mask_ind])
        masked_image = np.where(closest_mask, data, np.nan)

        return masked_image, closest_mask, merged_other_masks
