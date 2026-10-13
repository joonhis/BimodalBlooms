#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np


def identify_bloom_regions(chl_clim, depth, depth_threshold=2000):
    """
    Identify grid cells exhibiting spring blooms, fall blooms,
    and both spring and fall blooms.

    Parameters
    ----------
    chl_clim : np.ndarray
        Monthly climatology of surface chlorophyll-a.
        Expected shape: (12, ny, nx)

    depth : np.ndarray
        Bathymetry in meters.
        Expected shape: (ny, nx)

    depth_threshold : float, optional
        Minimum water depth used to exclude coastal regions.
        Default is 2000 m.

    Returns
    -------
    spring_bloom_mask : np.ndarray
        Boolean mask for grid cells exhibiting a spring bloom.

    fall_bloom_mask : np.ndarray
        Boolean mask for grid cells exhibiting a fall bloom.

    bimodal_bloom_mask : np.ndarray
        Boolean mask for grid cells exhibiting both spring and fall blooms.

    spring_idx : tuple
        Indices of spring bloom grid cells.

    fall_idx : tuple
        Indices of fall bloom grid cells.

    bimodal_idx : tuple
        Indices of grid cells exhibiting both spring and fall blooms.
    """

    # -------------------------------------------------------------------------
    # 1. Seasonal median Chl-a
    # -------------------------------------------------------------------------

    spring_chl = np.nanmedian(
        chl_clim[2:5, ...],
        axis=0
    )  # March-May

    summer_chl = np.nanmedian(
        chl_clim[5:8, ...],
        axis=0
    )  # June-August

    fall_chl = np.nanmedian(
        chl_clim[8:11, ...],
        axis=0
    )  # September-November

    winter_chl = np.nanmedian(
        chl_clim[[11, 0, 1], ...],
        axis=0
    )  # December-February


    # -------------------------------------------------------------------------
    # 2. Open-ocean mask
    # -------------------------------------------------------------------------

    open_ocean = depth > depth_threshold


    # -------------------------------------------------------------------------
    # 3. Bloom identification
    # -------------------------------------------------------------------------

    # Spring bloom:
    # median(MAM) > median(DJF)
    spring_bloom_mask = (
        (spring_chl > winter_chl)
        & open_ocean
    )

    # Fall bloom:
    # median(SON) > median(JJA)
    fall_bloom_mask = (
        (fall_chl > summer_chl)
        & open_ocean
    )

    # Bimodal bloom:
    # both spring and fall bloom criteria are satisfied
    bimodal_bloom_mask = (
        spring_bloom_mask
        & fall_bloom_mask
    )


    # -------------------------------------------------------------------------
    # 4. Grid-cell indices
    # -------------------------------------------------------------------------

    spring_idx = np.where(spring_bloom_mask)
    fall_idx = np.where(fall_bloom_mask)
    bimodal_idx = np.where(bimodal_bloom_mask)


    return (
        spring_bloom_mask,
        fall_bloom_mask,
        bimodal_bloom_mask,
        spring_idx,
        fall_idx,
        bimodal_idx,
    )


# In[ ]:




