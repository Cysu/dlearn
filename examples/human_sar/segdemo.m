for i = 5300:8500
    I = images{i};
    S = segmentations{i};
    imshow(I); hold on;
    h = imshow(uint8(S >= 0.5), [0,0,0; 1,0.5216,0.1059]); hold off;
    set(h, 'AlphaData', (S >= 0.5) * 0.7);
    pause;
end
    