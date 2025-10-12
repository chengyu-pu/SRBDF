# SRBDF
Spatiotemporal Radar Backscatter Data Fusion (SRBDF) framework for monitoring surface melt on Antarctic ice shelves.
This repository provides the codes for the SRBDF framework. The framework integrates Sentinel-1 and ASCAT backscatter data to generate high-resolution daily surface melt products over Antarctic ice shelves.

## 1. Introduction
The SRBDF framework aims to enhance the spatial and temporal resolution of radar backscatter observations for accurate monitoring of Antarctic surface melt.

## 2. Preprocessing (Google Earth Engine)

All Sentine-1 SAR preprocessing steps were implemented in Google Earth Engine (GEE), including:
(1)Refined Lee filtering for speckle noise reduction;
(2)Cosine-based incidence angle normalization(normalized to 40° reference angle);
(3)Image clipping and export.

## 3. Spatiotemporal Radar Backscatter Data Fusion
The SRBDF framework integrates Sentinel-1 and ASCAT C-band backscatter observations through a spatiotemporal fusion process.  
The method follows the principles of the FSDAF (Flexible Spatiotemporal Data Fusion) algorithm, adapted for active microwave data. The FSDAF implementation was based on the open-source codes developed by Dr. Xiaolin Zhu, with modifications to support radar datasets.

## 4. Visualization and validation (Python)
Python codes are provided for data visualization and performance evaluation of the SRBDF.

## 5. Data Sources
Sentinel-1 GRD data: Copernicus Data Space Ecosystem (https://browser.dataspace.copernicus.eu/);
ASCAT data: Brigham Young University (BYU) (https://ftp.scp.byu.edu/data/ascat/)).

## 6. Contact
For questions or collaboration inquiries, please contact:  
**Chengyu Pu** – School of Geospatial Engineering and Science, Sun Yat-sen University  
Email: puchy3@mail2.sysu.edu.cn
