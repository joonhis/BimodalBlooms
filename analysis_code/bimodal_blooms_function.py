#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


def identify_bimodal_bloom(chl_clim, depth):
    """
    Identify grid cells exhibiting both spring and fall phytoplankton blooms.

    Parameters
    ----------
    chl_clim : np.ndarray
        Monthly climatology of surface chlorophyll-a.
        Expected shape: (12, ny, nx)

    depth : np.ndarray
        Bathymetry in meters.
        Expected shape: (ny, nx)

    Returns
    -------
    bimodal_bloom_mask : np.ndarray
        Boolean mask indicating grid cells satisfying all criteria.

    lat_idx, lon_idx : np.ndarray
        Indices of grid cells satisfying the bimodal bloom criteria.
    """

    # Seasonal median Chl-a
    spring_chl = np.nanmedian(chl_clim[2:5,  ...], axis=0)       # MAM
    summer_chl = np.nanmedian(chl_clim[5:8,  ...], axis=0)       # JJA
    fall_chl   = np.nanmedian(chl_clim[8:11,  ...], axis=0)      # SON
    winter_chl = np.nanmedian(chl_clim[[11, 0, 1],  ...], axis=0)  # DJF

    # Bloom criteria
    spring_bloom = spring_chl > winter_chl
    fall_bloom   = fall_chl > summer_chl

    # Exclude coastal regions shallower than 2000 m
    open_ocean = depth > 2000

    # Grid cells satisfying all criteria
    bimodal_bloom_mask = (
        spring_bloom
        & fall_bloom
        & open_ocean
    )

    # Selected grid-cell indices
    lat_idx, lon_idx = np.where(bimodal_bloom_mask)

    return bimodal_bloom_mask, lat_idx, lon_idx


# In[ ]:




