Introduction
============
These datasets are formatted for human attribute classification.

Specification
=============
Each dataset is stored in MAT-File format.

    dataset_name.mat
        images : M*1 cells, each is an image
        attributes : M*1 cells, each is a D*1 logical vector
        attribute_names : D*1 cells, each is a string
        segmentations : M*1 cells, each is a segmentation map
        segmentation_names : D*2 cells, each is a pair of segmentation map pixel
            value and its semantic meaning, including background
        colormap : The colormap used to show segmentation map

Datasets
========
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Size ``M``</th>
            <th>Size ``D``</th>
            <th>Comment</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CUHK_SAR</td>
            <td>2368</td>
            <td>99</td>
            <td>Every consecutive pair of images belongs to the same person but
                is captured under different camera views.</td>
        </tr>
    </tbody>
<table>