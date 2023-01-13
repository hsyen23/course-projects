clear; clc;
load('Prior_1.mat');
load('Alpha.mat');
load('TrainingSamplesDCT_subsets_8.mat');
cheetah = imread('cheetah.bmp');
cheetah=double(cheetah)/255;
% store data into list
Data_FG_list = {D1_FG, D2_FG, D3_FG, D4_FG};
Data_BG_list = {D1_BG, D2_BG, D3_BG, D4_BG};


for strategy_idx = 1:2

if (strategy_idx == 1)
    load('Prior_1.mat');
else
    load('Prior_2.mat');
end

for data_index = 1:4

Data_FG = cell2mat(Data_FG_list(data_index));
Data_BG = cell2mat(Data_BG_list(data_index));

% ML mu for cheetah nad grass
D1_FG_mean = mean(Data_FG);
D1_BG_mean = mean(Data_BG);

% Covariance for FG and BG
D1_FG_covariance = cov(Data_FG) * ((length(Data_FG)-1)) / (length(Data_FG));
D1_BG_covariance = cov(Data_BG) * ((length(Data_BG)-1)) / (length(Data_BG));

% here is for problem a)
% loop start
for ii = 1:9

% Cov0
cov0 = diag(alpha(ii) * W0);

% cheetah_mu_n
a1_FG = length(Data_FG) * cov0 / (D1_FG_covariance + length(Data_FG)*cov0);
a2_FG = D1_FG_covariance / (D1_FG_covariance + length(Data_FG) * cov0);
cheetah_mu_n = (a1_FG * D1_FG_mean.' + a2_FG * mu0_FG.').';

% grass_mu_n
a1_BG = length(Data_BG) * cov0 / (D1_BG_covariance + length(Data_BG)*cov0);
a2_BG = D1_BG_covariance / (D1_BG_covariance + length(Data_BG) * cov0);
grass_mu_n = (  a1_BG * D1_BG_mean.' + a2_BG * mu0_BG.').';

% cheetah_covariance_n
cheetah_covariance_n = D1_FG_covariance * cov0 / (D1_FG_covariance + length(Data_FG) * cov0);
% grass_covariance_n
grass_covariance_n = D1_BG_covariance * cov0 / (D1_BG_covariance + length(Data_BG) * cov0);

% posteria_cheetah_covariance
posteria_cheetah_covariance = cheetah_covariance_n + D1_FG_covariance;
% posteria_grass_covariance
posteria_grass_covariance = grass_covariance_n + D1_BG_covariance;

% prior probability for class
p_cheetah = size(Data_FG,1) / (size(Data_FG,1) + size(Data_BG,1));
p_grass = size(Data_BG,1) / (size(Data_FG,1) + size(Data_BG,1));

% start to classify
row_size = size(cheetah, 1);
column_size = size(cheetah, 2);
A = zeros(row_size, column_size);

% using 8 * 8 blocks to represent the left top pixel
for rows = 1 : row_size - 8 + 1
    for columns = 1 : column_size - 8 + 1
        block = cheetah(rows:rows+7, columns:columns+7);
        block = dct2(block);
        x_value = expand_zigzag(block);
        % calculate P(1,x) and P(0,x), find bigger one

        P_0 = (-0.5*(x_value - grass_mu_n)/ posteria_grass_covariance * (x_value - grass_mu_n).') - log(sqrt(det(posteria_grass_covariance)*(2*pi)^64)) + log(p_grass);

        P_1 = (-0.5*(x_value - cheetah_mu_n)/ posteria_cheetah_covariance * (x_value - cheetah_mu_n).') - log(sqrt(det(posteria_cheetah_covariance)*(2*pi)^64)) + log(p_cheetah);
        if (P_0 >= P_1)
            A(rows, columns) = 0;
        else
            A(rows, columns) = 1;
        end
    end
end

% calculate error
% load cheetah mask.bmp
truth = imread("cheetah_mask.bmp");
truth = double(truth/255);
err = truth - A;
err = abs(err);
probability_error = sum(err,'all') / (size(A,1)*size(A,2));

storage(ii) = probability_error;
end
% loop end

% here is for problem b)
% start to classify for ML
row_size = size(cheetah, 1);
column_size = size(cheetah, 2);
A = zeros(row_size, column_size);

% using 8 * 8 blocks to represent the left top pixel
for rows = 1 : row_size - 8 + 1
    for columns = 1 : column_size - 8 + 1
        block = cheetah(rows:rows+7, columns:columns+7);
        block = dct2(block);
        x_value = expand_zigzag(block);
        % calculate P(1|x) and P(0|x), find bigger one

        P_0 = (-0.5*(x_value - D1_BG_mean)/ D1_BG_covariance * (x_value - D1_BG_mean).') - log(sqrt(det(D1_BG_covariance)*(2*pi)^64)) + log(p_grass);

        P_1 = (-0.5*(x_value - D1_FG_mean)/ D1_FG_covariance * (x_value - D1_FG_mean).') - log(sqrt(det(D1_FG_covariance)*(2*pi)^64)) + log(p_cheetah);
        if (P_0 >= P_1)
            A(rows, columns) = 0;
        else
            A(rows, columns) = 1;
        end
    end
end
err = truth - A;
err = abs(err);
probability_error = sum(err,'all') / (size(A,1)*size(A,2));



% here is for problem c)
% loop start
for ii = 1:9

% Cov0
cov0 = diag(alpha(ii) * W0);

% cheetah_mu_n
a1_FG = cov0 / (cov0 + (1/length(Data_FG))* D1_FG_covariance);
a2_FG = (1/length(Data_FG)) * D1_FG_covariance / (cov0 + (1/length(Data_FG))* D1_FG_covariance);
cheetah_mu_n = (a1_FG * D1_FG_mean.' + a2_FG * mu0_FG.').';

% grass_mu_n
a1_BG = cov0 / (cov0 + (1/length(Data_BG))* D1_BG_covariance);
a2_BG = (1/length(Data_BG)) * D1_BG_covariance / (cov0 + (1/length(Data_BG))* D1_BG_covariance);
grass_mu_n = (a1_BG * D1_BG_mean.' + a2_BG * mu0_BG.').';

% cheetah_covariance_n
cheetah_covariance_n = cov0 / (cov0 + (1/length(Data_FG))* D1_FG_covariance) * ((1/length(Data_FG))*D1_FG_covariance);
% grass_covariance_n
grass_covariance_n = cov0 / (cov0 + (1/length(Data_BG))* D1_BG_covariance) * ((1/length(Data_BG))*D1_BG_covariance);

% posteria_cheetah_covariance
posteria_cheetah_covariance = cheetah_covariance_n + D1_FG_covariance;
% posteria_grass_covariance
posteria_grass_covariance = grass_covariance_n + D1_BG_covariance;

p_cheetah = size(Data_FG,1) / (size(Data_FG,1) + size(Data_BG,1));
p_grass = size(Data_BG,1) / (size(Data_FG,1) + size(Data_BG,1));

% start to classify
row_size = size(cheetah, 1);
column_size = size(cheetah, 2);
A = zeros(row_size, column_size);

% using 8 * 8 blocks to represent the left top pixel
for rows = 1 : row_size - 8 + 1
    for columns = 1 : column_size - 8 + 1
        block = cheetah(rows:rows+7, columns:columns+7);
        block = dct2(block);
        x_value = expand_zigzag(block);
        % calculate P(1|x) and P(0|x), find bigger one

        P_0 = (-0.5*(x_value - grass_mu_n)/ D1_BG_covariance * (x_value - grass_mu_n).') - log(sqrt(det(D1_BG_covariance)*(2*pi)^64)) + log(p_grass);

        P_1 = (-0.5*(x_value - cheetah_mu_n)/ D1_FG_covariance * (x_value - cheetah_mu_n).') - log(sqrt(det(D1_FG_covariance)*(2*pi)^64)) + log(p_cheetah);
        if (P_0 >= P_1)
            A(rows, columns) = 0;
        else
            A(rows, columns) = 1;
        end
    end
end
% error
% load cheetah mask.bmp
truth = imread("cheetah_mask.bmp");
truth = double(truth/255);

err = truth - A;
err = abs(err);
probability_error = sum(err,'all') / (size(A,1)*size(A,2));
storage2(ii) = probability_error;
% loop end
end

%  plot error
figure();
semilogx(alpha,storage);
hold on;
semilogx(alpha,storage2);
yline(probability_error);
legend('Bayesian','MAP approximation','ML')
txt = "dataset " + int2str(data_index) +" strategy " + int2str(strategy_idx);
title(txt);
hold off;
end

end