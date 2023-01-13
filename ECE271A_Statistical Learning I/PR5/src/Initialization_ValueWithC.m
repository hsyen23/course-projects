function [p_pi, mu, covariance] = Initialization_ValueWithC(data,c)
    p_pi = rand(1, c);
    p_pi = p_pi / sum(p_pi); % initial pi

    % pick one data from dataset as initial value for mu

    mu = zeros(c, 64);
    for counter = 1 : c
      mu(counter,:) = data(randi(size(data,1)),:); 
    end

    covariance = zeros(64, 64, c);
    for counter = 1 : c
        covariance(:,:,counter) = (rand(1, 64)).* eye(64);
    end

end