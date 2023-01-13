function [new_pi, new_mu, new_cov, time] = runEMwithC(data, pre_pi, pre_mu, pre_cov, c)

for time = 1 : 1000
% create h
h = zeros(size(data,1), c);
BDRjoint=zeros(size(data,1), c);
% E-step
% calculate h
for i = 1 : size(data,1)
    for j = 1:c
        BDRjoint(i,j) = mvnpdf(data(i,:),pre_mu(j,:),pre_cov(:,:,j)) * pre_pi(j);
    end
    h_denominator = sum(BDRjoint(i,:));
    h(i,:) = BDRjoint(i,:) / h_denominator;
end
BDRlikely(time) = sum( log( sum(BDRjoint,2) )  );

% M-step
% update paremeters
new_pi = sum(h) / size(data,1);

%update for mu
new_mu = zeros(c, 64);
for j = 1 : c

    tem_data = data;
    for i = 1 : size(data, 1)
        tem_data(i,:) = tem_data(i,:) * h(i,j);
    end
    new_mu(j,:) = sum(tem_data) / sum(h(:,j));
end

% update for covariance
new_cov = zeros(64,64,c);

for j = 1 : c
    new_cov(:,:,j) = diag(diag(( (data-pre_mu(j,:))' .*h(:,j)'*(data-pre_mu(j,:))./sum(h(:,j),1))+0.0001));
end

pre_pi = new_pi; 
pre_mu = new_mu;
pre_cov = new_cov;

if (time > 1)
    if (abs(BDRlikely(time) - BDRlikely(time-1))<0.001)
        break;
    end
end

end

end