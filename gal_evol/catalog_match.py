import numpy as np
from astropy.coordinates import SkyCoord

class Match_Gal():
    """
    Match_Gal
    A class for processing and analyzing galaxy segmentation maps, particularly focusing on 
    high-redshift galaxies. This class provides methods to combine segments of high-redshift 
    galaxies into a single mask and identify background segments not included in the high-redshift mask.
    Methods:
    --------
    - combine_highz_segments(segment, wcs, galaxy_ra, galaxy_dec, galaxy_redshifts, z_lim=3.08):
        background segments not included in the high-redshift mask. Processes a segmentation 
        map to identify and combine regions corresponding to galaxies with redshifts greater 
        than a specified limit.
    - combine_highz_segments_arr(good_indices, objectID_arr, bad_deblended_objectIDs, catalog, 
      catalog_matcher, wcs_arr, segments_arr, deblended_segments_arr, 
      use_3sigma_redshift_confidence=True, z_lim=3.08):
        Processes multiple segmentation maps and catalogs to generate masks for high-redshift 
        galaxies and background segments. Handles arrays of segmentation maps and associated 
        metadata, allowing for batch processing of galaxy data.
    """
    
    def __init__(self):
        pass
        
    def combine_highz_segments(self, segment, wcs, galaxy_ra, galaxy_dec, galaxy_redshifts, z_lim=3.08, do_force_segment=False, force_segmentation_values=[]):
        """
        Combines segments of high-redshift galaxies into a single mask and identifies 
        background segments not included in the high-redshift mask.

        This method processes a segmentation map to identify and combine regions 
        corresponding to galaxies with redshifts greater than a specified limit. 
        It also creates a mask for all other segments not included in the high-redshift mask.

        Parameters:
        -----------
        segment : Segment
            A segmentation map where each pixel is labeled with an integer corresponding 
            to a specific segment.
        wcs : WCS
            World Coordinate System (WCS) object for converting between world and pixel coordinates.
        galaxy_ra : array-like
            Right Ascension (RA) values of galaxies in degrees.
        galaxy_dec : array-like
            Declination (Dec) values of galaxies in degrees.
        galaxy_redshifts : array-like
            Redshift values of galaxies.
        z_lim : float, optional
            Redshift threshold above which galaxies are considered high-redshift. Defaults to 3.08.
        do_force_segment : bool, optional
            If True, forces specific segments to be included in the high-redshift mask. Defaults to False.
        force_segmentation_values : list, optional
            List of segment labels to forcefully include in the high-redshift mask. Defaults to an empty list.

        Returns:
        --------
        tuple
            - combined_segments (ndarray): A boolean mask where regions corresponding 
              to high-redshift galaxies are marked as True.
            - bg_segments_mask (ndarray): A boolean mask where regions corresponding 
              to background segments (not part of high-redshift galaxies) are marked as True.
        """
        # Convert galaxy RA/Dec to pixel coordinates using WCS
        galaxies_coords = SkyCoord(ra=galaxy_ra, dec=galaxy_dec, unit="deg")
        galaxy_pixels = wcs.world_to_pixel(galaxies_coords)
        galaxy_x, galaxy_y = galaxy_pixels[0], galaxy_pixels[1]
        # Identify galaxies with redshift > z_lim
        high_z_indices = np.where(galaxy_redshifts > z_lim)[0]
        high_z_x = galaxy_x[high_z_indices]
        high_z_y = galaxy_y[high_z_indices]
        # Initialise a combined mask for high-z galaxies
        combined_segments = np.zeros_like(segment.data, dtype=bool)
        # Loop through all high-redshift galaxies
        for x, y in zip(high_z_x, high_z_y):
            # Convert pixel coordinates to integer
            x = int(round(x))
            y = int(round(y))
            # Ensure the coordinates are within image bounds
            if 0 <= x < combined_segments.shape[1] and 0 <= y < combined_segments.shape[0]:
                # Find the segment label corresponding to this pixel
                label = segment.data[y, x]
                if label > 0:  # Exclude background (label 0)
                    # Add the current segment to the combined high-z mask
                    combined_segments |= (segment.data == label)
        if do_force_segment == True:
            for i in range(len(force_segmentation_values)):
                combined_segments |= (segment.data == force_segmentation_values[i])
        # Create the masked image by setting high-z galaxy regions to NaN
        #combined_segments_data = np.where(combined_segments, np.nan, data.data)
        # Create a mask for all other segments not included in the high-z mask
        bg_segments_mask = (segment.data > 0) & ~combined_segments
        return combined_segments, bg_segments_mask

    def combine_highz_segments_arr(self, good_indices, objectID_arr,
                                   bad_deblended_objectIDs, catalog, catalog_matcher,
                                   wcs_arr, segments_arr, deblended_segments_arr,
                                   use_3sigma_redshift_confidence=True, force_segmentation_indices=[],
                                   force_segmentation_values=[], z_lim=3.08):
        """
        Combines high-redshift galaxy segmentation masks for a set of objects.
        This method processes a list of objects, extracting their photometric redshifts, 
        coordinates, and segmentation masks to generate combined segmentation masks 
        for high-redshift galaxies. It also handles cases where objects are flagged 
        as poorly deblended.
            good_indices (list): Indices of objects to process.
            objectID_arr (list): Array of object IDs corresponding to the input objects.
            bad_deblended_objectIDs (list): List of object IDs flagged as poorly deblended.
            catalog (list): Catalog data containing photometric and positional information.
            catalog_matcher (list): Mapping between input indices and catalog entries.
            wcs_arr (list): Array of WCS (World Coordinate System) objects for each object.
            segments_arr (list): Array of segmentation masks for the objects.
            deblended_segments_arr (list): Array of deblended segmentation masks for the objects.
            use_3sigma_redshift_confidence (bool, optional): If True, uses the lower-limit 
                (99 percent confidence interval) photometric redshift. Defaults to True.
            z_lim (float, optional): Redshift limit to classify high-redshift galaxies. 
                Defaults to 3.08.
            tuple: A tuple containing:
                - combined_segment_mask_arr (list): List of combined segmentation masks 
                  for high-redshift galaxies.
                - bg_segments_mask_arr (list): List of background segmentation masks.
        """
        combined_segment_mask_arr, bg_segments_mask_arr = [], [] 
        for ind in good_indices:

            try:
                # Extract RA, Dec, and redshift for the source
                if use_3sigma_redshift_confidence:
                    print('using lower-limit (99 percent confidence interval) photometric redshift')
                    phot_zz   = catalog[-1].data[catalog_matcher[ind][0]]['EAZY_l99']
                else:
                    print('using mean photometric redshift')
                    phot_zz = catalog[-1].data[catalog_matcher[ind][0]]['EAZY_z_a']
                
                ra_s, dec_s = catalog[2].data[catalog_matcher[ind][0]]['ra'], catalog[2].data[catalog_matcher[ind][0]]['dec']
                # Select the appropriate segmentation mask
                if objectID_arr[ind] not in bad_deblended_objectIDs:
                    segment = deblended_segments_arr[ind]
                else:
                    segment = segments_arr[ind]
                # Generate masks for high-z galaxies
                if ind in force_segmentation_indices:
                    print('using forced segmentation', ind)
                    combined_segment_mask, bg_segments_mask = self.combine_highz_segments(
                        segment=segment,
                        wcs=wcs_arr[ind].wcs,
                        galaxy_ra=ra_s,  # Ensure RA is an array
                        galaxy_dec=dec_s,  # Ensure Dec is an array
                        galaxy_redshifts=phot_zz,  # Ensure redshifts are an array
                        z_lim=z_lim, 
                        do_force_segment=True,
                        force_segmentation_values=force_segmentation_values)
                    plt.imshow(combined_segment_mask)
                    combined_segment_mask_arr.append(combined_segment_mask)
                    bg_segments_mask_arr.append(bg_segments_mask)
                else:
                    combined_segment_mask, bg_segments_mask = self.combine_highz_segments(
                        segment=segment,
                        wcs=wcs_arr[ind].wcs,
                        galaxy_ra=ra_s,  # Ensure RA is an array
                        galaxy_dec=dec_s,  # Ensure Dec is an array
                        galaxy_redshifts=phot_zz,  # Ensure redshifts are an array
                        z_lim=z_lim, 
                        do_force_segment=False)
                    combined_segment_mask_arr.append(combined_segment_mask)
                    bg_segments_mask_arr.append(bg_segments_mask)

            except Exception as e:
                print(f"Error processing index: {e}")
                continue
        return combined_segment_mask_arr, bg_segments_mask_arr