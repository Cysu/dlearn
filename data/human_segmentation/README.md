Introduction
============
These datasets are formatted for human segmenation.

Specification
=============
Each dataset is stored in MAT-File format.

    dataset_name.mat
        images : M*1 cells, each is an image
        segmentations : M*1 cells, each is a segmentation map
        segmentation_names : D*2 cells, each is a pair of segmentation map pixel value and its semantic meaning, including background
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
    </tbody>
<table>