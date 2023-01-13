clear;clc;
% load trainging sample and image
load('TrainingSamplesDCT_8_new.mat');
cheetah = imread('cheetah.bmp');
cheetah=double(cheetah)/255;

% create 5 initial value for BG and FG and then run EM
%% 1th
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_BG);
[BG_pi_1, BG_mu_1, BG_cov_1] = runEM(TrainsampleDCT_BG, p_pi, mu, covariance);
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_FG);
[FG_pi_1, FG_mu_1, FG_cov_1] = runEM(TrainsampleDCT_FG, p_pi, mu, covariance);
%% 2th
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_BG);
[BG_pi_2, BG_mu_2, BG_cov_2] = runEM(TrainsampleDCT_BG, p_pi, mu, covariance);
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_FG);
[FG_pi_2, FG_mu_2, FG_cov_2] = runEM(TrainsampleDCT_FG, p_pi, mu, covariance);
%% 3th
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_BG);
[BG_pi_3, BG_mu_3, BG_cov_3] = runEM(TrainsampleDCT_BG, p_pi, mu, covariance);
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_FG);
[FG_pi_3, FG_mu_3, FG_cov_3] = runEM(TrainsampleDCT_FG, p_pi, mu, covariance);
%% 4th
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_BG);
[BG_pi_4, BG_mu_4, BG_cov_4] = runEM(TrainsampleDCT_BG, p_pi, mu, covariance);
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_FG);
[FG_pi_4, FG_mu_4, FG_cov_4] = runEM(TrainsampleDCT_FG, p_pi, mu, covariance);
%% 5th
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_BG);
[BG_pi_5, BG_mu_5, BG_cov_5] = runEM(TrainsampleDCT_BG, p_pi, mu, covariance);
[p_pi, mu, covariance] = Initialization_Value(TrainsampleDCT_FG);
[FG_pi_5, FG_mu_5, FG_cov_5] = runEM(TrainsampleDCT_FG, p_pi, mu, covariance);
%% combine them
FG_pi_set = {FG_pi_1,FG_pi_2,FG_pi_3,FG_pi_4,FG_pi_5};
BG_pi_set = {BG_pi_1,BG_pi_2,BG_pi_3,BG_pi_4,BG_pi_5};
FG_mu_set = {FG_mu_1,FG_mu_2,FG_mu_3,FG_mu_4,FG_mu_5};
BG_mu_set = {BG_mu_1,BG_mu_2,BG_mu_3,BG_mu_4,BG_mu_5};
FG_cov_set = {FG_cov_1,FG_cov_2,FG_cov_3,FG_cov_4,FG_cov_5};
BG_cov_set = {BG_cov_1,BG_cov_2,BG_cov_3,BG_cov_4,BG_cov_5};
%%
dimension_set = [1 2 4 8 16 32 64];

for fixed_FG = 1:5
        FG_pi = cell2mat(FG_pi_set(fixed_FG));
        FG_mu = cell2mat(FG_mu_set(fixed_FG));
        FG_cov = cell2mat(FG_cov_set(fixed_FG));
        figure(fixed_FG);
        txt = "FG mixture_ " + int2str(fixed_FG) +" comparision ";
        title(txt);
    for fix_BG = 1:5
        BG_pi = cell2mat(BG_pi_set(fix_BG));
        BG_mu = cell2mat(BG_mu_set(fix_BG));
        BG_cov = cell2mat(BG_cov_set(fix_BG));
%%
storage = zeros(1,7);
for di_counter = 1 : 7

d = dimension_set(di_counter);

% calculate prior probabilities of cheetah and grass
pixel_total_count = size(TrainsampleDCT_FG, 1) + size(TrainsampleDCT_BG, 1);
prior_Pcheetah = size(TrainsampleDCT_FG, 1) / pixel_total_count;
prior_Pgrass = size(TrainsampleDCT_BG, 1) / pixel_total_count;

% create output mask image array
row_size = size(cheetah, 1);
column_size = size(cheetah, 2);
A_64 = zeros(row_size, column_size);

% using 8 * 8 blocks to represent the left top pixel
for rows = 1 : row_size - 8 + 1
    for columns = 1 : column_size - 8 + 1
        block = cheetah(rows:rows+7, columns:columns+7);
        block = dct2(block);
        x = expand_zigzag(block);
        % for FG and BG
        p_FG = 0;
        for counter = 1 : 8
            p_FG = p_FG + mvnpdf(x(1:d), FG_mu(counter,1:d), FG_cov(1:d,1:d,counter)) * FG_pi(counter);
        end
        p_FG = p_FG * prior_Pcheetah;
        
        p_BG = 0;
        for counter = 1 : 8
            p_BG = p_BG + mvnpdf(x(1:d), BG_mu(counter,1:d), BG_cov(1:d,1:d,counter)) * BG_pi(counter);
        end
        p_BG = p_BG * prior_Pgrass;

        if (p_BG > p_FG)
            A_64(rows, columns) = 0;
        else
            A_64(rows, columns) = 1;
        end
    end
end

%figure();
%imagesc(A_64);
%colormap(gray(255));
%% error
% load cheetah mask.bmp
truth = imread("cheetah_mask.bmp");
% calculate last meaningful index of row and column
last_row = size(cheetah, 1) - 8 + 1;
last_column = size(cheetah, 2) - 8 + 1;
% error for 64d
truth = double(truth(1 : last_row, 1 : last_column) / 255);
A_64 = A_64(1 : last_row, 1 : last_column);
err = truth - A_64;
err = abs(err);
probability_error_64d = sum(err,'all') / (last_row*last_column);
storage(di_counter) = probability_error_64d;
end
plot(dimension_set, storage);
hold on;
    end
    hold off;
    legend('BG(1)','BG(2)','BG(3)','BG(4)','BG(5)');
end

%% run for different C
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_BG,1);
[BG_pi_1, BG_mu_1, BG_cov_1] = runEMwithC(TrainsampleDCT_BG, p_pi, mu, covariance, 1);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_BG,2);
[BG_pi_2, BG_mu_2, BG_cov_2] = runEMwithC(TrainsampleDCT_BG, p_pi, mu, covariance, 2);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_BG,4);
[BG_pi_4, BG_mu_4, BG_cov_4] = runEMwithC(TrainsampleDCT_BG, p_pi, mu, covariance, 4);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_BG,8);
[BG_pi_8, BG_mu_8, BG_cov_8] = runEMwithC(TrainsampleDCT_BG, p_pi, mu, covariance, 8);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_BG,16);
[BG_pi_16, BG_mu_16, BG_cov_16] = runEMwithC(TrainsampleDCT_BG, p_pi, mu, covariance, 16);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_BG,32);
[BG_pi_32, BG_mu_32, BG_cov_32] = runEMwithC(TrainsampleDCT_BG, p_pi, mu, covariance, 32);

%% 
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_FG, 1);
[FG_pi_1, FG_mu_1, FG_cov_1] = runEMwithC(TrainsampleDCT_FG, p_pi, mu, covariance, 1);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_FG, 2);
[FG_pi_2, FG_mu_2, FG_cov_2] = runEMwithC(TrainsampleDCT_FG, p_pi, mu, covariance, 2);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_FG, 4);
[FG_pi_4, FG_mu_4, FG_cov_4] = runEMwithC(TrainsampleDCT_FG, p_pi, mu, covariance, 4);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_FG, 8);
[FG_pi_8, FG_mu_8, FG_cov_8] = runEMwithC(TrainsampleDCT_FG, p_pi, mu, covariance, 8);
%%
clc;
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_FG, 16);
[FG_pi_16, FG_mu_16, FG_cov_16] = runEMwithC(TrainsampleDCT_FG, p_pi, mu, covariance, 16);
%%
[p_pi, mu, covariance] = Initialization_ValueWithC(TrainsampleDCT_FG, 32);
[FG_pi_32, FG_mu_32, FG_cov_32] = runEMwithC(TrainsampleDCT_FG, p_pi, mu, covariance, 32);

%% combine them
FG_pi_set = {FG_pi_1,FG_pi_2,FG_pi_4,FG_pi_8,FG_pi_16,FG_pi_32};
BG_pi_set = {BG_pi_1,BG_pi_2,BG_pi_4,BG_pi_8,BG_pi_16,BG_pi_32};
FG_mu_set = {FG_mu_1,FG_mu_2,FG_mu_4,FG_mu_8,FG_mu_16,FG_mu_32};
BG_mu_set = {BG_mu_1,BG_mu_2,BG_mu_4,BG_mu_8,BG_mu_16,BG_mu_32};
FG_cov_set = {FG_cov_1,FG_cov_2,FG_cov_4,FG_cov_8,FG_cov_16,FG_cov_32};
BG_cov_set = {BG_cov_1,BG_cov_2,BG_cov_4,BG_cov_8,BG_cov_16,BG_cov_32};
%%
C_list = [1 2 4 8 16 32];
figure();

for c_choosen = 1 : 6
c = C_list(c_choosen);

FG_pi = cell2mat(FG_pi_set(c_choosen));
FG_mu = cell2mat(FG_mu_set(c_choosen));
FG_cov = cell2mat(FG_cov_set(c_choosen));
BG_pi = cell2mat(BG_pi_set(c_choosen));
BG_mu = cell2mat(BG_mu_set(c_choosen));
BG_cov = cell2mat(BG_cov_set(c_choosen));
storage = zeros(1,7);
for di_counter = 1 : 7
d = dimension_set(di_counter);
% calculate prior probabilities of cheetah and grass
pixel_total_count = size(TrainsampleDCT_FG, 1) + size(TrainsampleDCT_BG, 1);
prior_Pcheetah = size(TrainsampleDCT_FG, 1) / pixel_total_count;
prior_Pgrass = size(TrainsampleDCT_BG, 1) / pixel_total_count;

% create output mask image array
row_size = size(cheetah, 1);
column_size = size(cheetah, 2);
A_64 = zeros(row_size, column_size);

% using 8 * 8 blocks to represent the left top pixel
for rows = 1 : row_size - 8 + 1
    for columns = 1 : column_size - 8 + 1
        block = cheetah(rows:rows+7, columns:columns+7);
        block = dct2(block);
        x = expand_zigzag(block);
        % for FG and BG
        p_FG = 0;
        for counter = 1 : c
            p_FG = p_FG + mvnpdf(x(1:d), FG_mu(counter,1:d), FG_cov(1:d,1:d,counter)) * FG_pi(counter);
        end
        p_FG = p_FG * prior_Pcheetah;
        
        p_BG = 0;
        for counter = 1 : c
            p_BG = p_BG + mvnpdf(x(1:d), BG_mu(counter,1:d), BG_cov(1:d,1:d,counter)) * BG_pi(counter);
        end
        p_BG = p_BG * prior_Pgrass;

        if (p_BG > p_FG)
            A_64(rows, columns) = 0;
        else
            A_64(rows, columns) = 1;
        end
    end
end
%figure();
%imagesc(A_64);
%colormap(gray(255));
%% error
% load cheetah mask.bmp
truth = imread("cheetah_mask.bmp");
% calculate last meaningful index of row and column
last_row = size(cheetah, 1) - 8 + 1;
last_column = size(cheetah, 2) - 8 + 1;
% error for 64d
truth = double(truth(1 : last_row, 1 : last_column) / 255);
A_64 = A_64(1 : last_row, 1 : last_column);
err = truth - A_64;
err = abs(err);
probability_error_64d = sum(err,'all') / (last_row*last_column);
storage(di_counter) = probability_error_64d;
end
plot(dimension_set, storage);
hold on;
end
hold off;
legend('C = 1','C = 2','C = 4','C = 8','C = 16','C = 32');
