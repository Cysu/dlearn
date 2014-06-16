function save_sar(i, images, attributes, segmentations, ind, attribute_names, colormap)

I = images{i};
A = attributes{i};
% S = segmentations{i};

a = sprintf('%04d', i);
for j = ind
    if A(j) == 1
        a = [a, ',', attribute_names{j}];
    end
end

% imwrite(I, fullfile('dataset_demo', [a, '.jpg']));
% imwrite(S, colormap, fullfile('dataset_demo', [a, '.png']));

imwrite(I, fullfile('dataset_demo_2', [a, '.jpg']));
% imwrite(S, colormap, fullfile('dataset_demo_2', [a, '.png']));
