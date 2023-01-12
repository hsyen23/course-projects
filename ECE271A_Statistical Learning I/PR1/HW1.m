clear; clc;
% load trainging sample and image
load('TrainingSamplesDCT_8.mat');
cheetah = imread('cheetah.bmp');
% calculate prior probabilities of cheetah and grass
pixel_total_count = size(TrainsampleDCT_FG, 1) + size(TrainsampleDCT_BG, 1);
prior_Pcheetah = size(TrainsampleDCT_FG, 1) / pixel_total_count;
prior_Pgrass = size(TrainsampleDCT_BG, 1) / pixel_total_count;
% calculate P(x|cheetah) and P(x|grass)
BG_data = zeros(1, 64); % set an empty array for stroring x variable frequency
FG_data = zeros(1, 64);
% using find_second_large function to get x variable
for row = 1 : size(TrainsampleDCT_BG,1)
    idx = find_second_large(TrainsampleDCT_BG(row,:));
    BG_data(idx) = BG_data(idx) + 1;
end

for row = 1 : size(TrainsampleDCT_FG,1)
    idx = find_second_large(TrainsampleDCT_FG(row,:));
    FG_data(idx) = FG_data(idx) + 1;
end
% convert cumulative data into probability
FG_data = FG_data / size(TrainsampleDCT_FG, 1);
BG_data = BG_data / size(TrainsampleDCT_BG, 1);
% display as histogram
subplot(1,2,1);
bar(BG_data);
title('P(x|grass)');
subplot(1,2,2);
bar(FG_data);
title('P(x|cheetah)');
% create a new matrix A for storing decision
row_size = size(cheetah, 1);
column_size = size(cheetah, 2);
A = zeros(row_size, column_size);
% using 8 * 8 blocks to represent the left top pixel
for rows = 1 : row_size - 8 + 1
    for columns = 1 : column_size - 8 + 1
        block = cheetah(rows:rows+7, columns:columns+7);
        block = dct2(block);
        % get X feature
        x_feature = find_second_large(expand_zigzag(block));
        % calculate P(1|x) and P(0|x), find bigger one
        P_0 = BG_data(x_feature) * prior_Pgrass;
        P_1 = FG_data(x_feature) * prior_Pcheetah;
        if (P_0 >= P_1)
            A(rows, columns) = 0;
        else
            A(rows, columns) = 1;
        end
    end
end
% display image
figure(2);
imagesc(A);
colormap(gray(255));
% load cheetah mask.bmp
truth = imread("cheetah_mask.bmp");
% calculate last meaningful index of row and column
last_row = row_size - 8 + 1;
last_column = column_size - 8 + 1;
% only take meaningful part
truth = double(truth(1 : last_row, 1 : last_column) / 255);
A = A(1 : last_row, 1 : last_column);
err = truth - A;
err = abs(err);
probability_error = sum(err,'all') / (last_row*last_column);