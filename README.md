# gal_evol

Tools for building galaxy image cutouts from JWST/HST mosaics, segmenting them,
masking contaminating neighbours, and saving the results to FITS. It was written
for studying high-redshift galaxies (for example EELGs and a matched control
sample from JADES GOODS-S), but the pieces work on any list of sky positions.

The idea is to take a target filter and a secondary filter, cut out a small
stamp around each source, find which segment belongs to the galaxy of interest,
replace everything else with noise, recentre, and write one FITS file per
filter pair.

## Install

The packaging files (`setup.py`, `setup.cfg`, `pyproject.toml`) are currently
empty, so there is no pip install yet. Clone the repo and add it to your path:

```bash
git clone https://github.com/himmng/gal_evol.git
cd gal_evol
export PYTHONPATH=$PWD:$PYTHONPATH
```

Dependencies (not yet pinned anywhere):

- numpy
- scipy
- astropy
- photutils (uses `make_2dgaussian_kernel`, `detect_sources`, `deblend_sources`, `Background2D`)
- matplotlib (for plotting in the example notebook)

## Modules

| Module | Class | Purpose |
| --- | --- | --- |
| `gal_evol/create_cutouts.py` | `Cutouts` | Make cutouts from an HDUList around RA/Dec, recentred on the local flux peak |
| `gal_evol/create_segments.py` | `Segmentation` | Background subtraction, convolution, source detection and deblending |
| `gal_evol/catalog_match.py` | `Match_Gal` | Build a high-z galaxy mask and a mask of all other segments using catalog redshifts |
| `gal_evol/background.py` | `Background_remove` | Fill masked regions with Gaussian noise matched to the unmasked pixels |
| `gal_evol/image_process.py` | `Center_Cutouts`, `Save_Data` | Recentre on the segment peak, pad or crop to a fixed size, save everything to FITS |

`gal_evol/utils.py` and `gal_evol/__init__.py` are empty placeholders.

### `Cutouts(header_index, size_arsec, tolerance_arcsec)`

Note that the keyword is spelled `size_arsec` in the current code.

- `create_cutouts(image, sky_position)` makes a cutout at an RA/Dec (degrees),
  finds the brightest pixel within `tolerance_arcsec` of the centre, then
  cuts again centred on that peak. Returns the cutout, the peak pixel
  position and the WCS.
- `process_cutout_array(sky_position_arr, target_filter_image, secondary_filter_image, avoid_target=False)`
  does this for a list of positions. The secondary filter is cut at the same
  pixel position as the target. Returns the target cutouts, secondary cutouts
  and an array of indices that failed. With `avoid_target=True` only the
  secondary cutouts and the error indices are returned.

### `Segmentation(use_median_background, box_size, filt_size, npixels_segmentation, npixels_deblend, sigma_clip, nlevels, contrast, gaussian_fwhm=2.5, gaussian_size=3)`

- `get_segmentation_with_deblend(image)` runs `Background2D`, subtracts the
  background, convolves with a Gaussian kernel, thresholds at the
  sigma-clipped standard deviation, and deblends. Returns the deblended map,
  the plain segmentation map, the background-subtracted data and the
  convolved data.
- `get_deblend_segmentation(convolved_image, segment_image)` deblends an
  existing segmentation map.
- `get_segmentations_arr(image_arr)` runs the above over a list of cutouts and
  returns the four lists plus an array of failed indices.
- `apply_closest_mask(data, segment)` picks the segment closest to the image
  centre. Returns the masked image (NaN outside the segment), the boolean mask
  of that segment, and a merged boolean mask of all other segments.

### `Match_Gal()`

- `combine_highz_segments(segment, wcs, galaxy_ra, galaxy_dec, galaxy_redshifts, z_lim=3.08, do_force_segment=False, force_segmentation_values=[])`
  converts catalog positions to pixels, takes every galaxy with redshift above
  `z_lim`, and merges the segments those galaxies fall in. Returns
  `(combined_segments, bg_segments_mask)` as boolean arrays. Specific segment
  labels can be forced into the mask.
- `combine_highz_segments_arr(...)` does the same for many cutouts using
  catalog tables. It reads `EAZY_l99` (lower 99 percent limit) or `EAZY_z_a`
  from the last catalog extension and `ra`/`dec` from extension 2. It picks the
  deblended map unless the object ID is listed in `bad_deblended_objectIDs`.

### `Background_remove()`

- `fill_masked_with_noise_combined(image, source_mask=None, background_mask=None)`
  replaces masked pixels with `Normal(mean, std)` drawn from the unmasked
  pixels. When a background mask is given it also returns a second image where
  only the background mask was filled.
- `process_images_with_masks(data_arrays, source_masks, background_masks, good_indices, filt=True)`
  runs it over lists and returns `(img_only_noise_list, img_with_source_list)`.

### `Center_Cutouts()` and `Save_Data(...)`

- `Center_Cutouts.center_cutouts(...)` finds the peak inside the segment, rolls
  the images so it sits at the centre, updates the WCS, crops to `npixels`,
  and fills NaNs with noise from the background stamp. `center_cutout_arr(...)`
  is the list version, and `pad_cutout_to_size(cutout, target_shape)` pads with
  NaN and fixes the WCS.
- `Save_Data(...).save_fits(sample_type, target_filter_name, secondary_filter_name, savepath, skip_target=False)`
  writes `{sample_type}_final_filters_{target}_{secondary}.fits` with these
  extensions: original images for both filters, segmentation masks, cutouts
  and backgrounds for both filters, one `WCS` extension per object, and one
  extension per field of the sample table. With `skip_target=True` only the
  secondary filter is written to `{sample_type}_final_filters_{secondary}.fits`.

## Typical workflow

`testing.ipynb` runs the full pipeline. In short:

```python
from astropy.io import fits
from gal_evol.create_cutouts import Cutouts
from gal_evol.create_segments import Segmentation
from gal_evol.catalog_match import Match_Gal
from gal_evol.background import Background_remove
from gal_evol.image_process import Center_Cutouts, Save_Data

f0 = fits.open("target_filter_drz.fits")
f1 = fits.open("secondary_filter_drz.fits")

# 1. cutouts around each sky position, recentred on the local peak
cutouts = Cutouts(header_index=1, size_arsec=5, tolerance_arcsec=0.1)
tgt, sec, err = cutouts.process_cutout_array(sky_position_arr, f0, f1)

# 2. segmentation and deblending
seg = Segmentation(use_median_background=True, box_size=8, filt_size=7,
                   npixels_segmentation=20, npixels_deblend=20,
                   nlevels=20, contrast=0.001, sigma_clip=3)
deblended, segmap, bgsub, convolved, err_ind = seg.get_segmentations_arr(tgt)

# 3. high-z mask and neighbour mask from catalog redshifts
match = Match_Gal()
highz_masks, bg_masks = match.combine_highz_segments_arr(
    good_indices, objectID_arr, bad_deblended_objectIDs, catalog,
    catalog_matcher, tgt, segmap, deblended)

# 4. fill neighbours with noise
noise_only, with_source = Background_remove().process_images_with_masks(
    tgt, highz_masks, bg_masks, good_indices)

# 5. recentre, crop, and save with Center_Cutouts and Save_Data
```

The notebook currently reads data from a hard coded local path
(`/media/himanshu/MyHDD/data_galaxy_evolution/data/`). Change `data_path` to
wherever your mosaics and catalogs live.

## Input data

The notebook expects:

- JADES NIRCam GOODS-S mosaics (`*_drz.fits`) for the filters you pick
- the JADES photometry catalog and NIRSpec line-flux catalogs
- a sample table (comma separated, header row) with at least `ra`, `dec`
  and `zspec` columns

## Known issues

These come from reading the code and are worth fixing:

- `catalog_match.py` calls `plt.imshow` in the forced-segmentation branch but
  never imports matplotlib, so that branch raises `NameError` (the surrounding
  `try/except` swallows it and the object is skipped).
- `create_segments.py`: `get_segmentation_with_deblend` can reference unset
  variables if an exception is raised, and `get_segmentations_arr` only
  appends results when a segmentation map exists, so the output lists can
  fall out of step with the input.
- `Cutouts.__init__` uses the misspelled argument `size_arsec`.
- Packaging files and `requirements.py` are empty.
- No tests besides the example notebook.
