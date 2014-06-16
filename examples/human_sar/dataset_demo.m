ind = [74,75,76,79,80,85,87,89,90,98,99];

% load('../../data/human_sar/CUHK_SAR.mat');
load('../../data/human_attribute/Mix.mat');
%%
for i =8000:length(images)
    I = images{i};
    A = attributes{i};
%     S = segmentations{i};
    imshow(I);
    
    a = sprintf('%04d', i);
    for j = ind
        if A(j) == 1
            a = [a, ',', attribute_names{j}];
        end
    end
    title(a);
    pause;    
end
