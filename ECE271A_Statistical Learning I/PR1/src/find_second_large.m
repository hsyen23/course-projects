function x_index = find_second_large(data)
    for index = 1 : size(data)
        arr = abs(data);
        x_value = max(arr(arr<max(arr)));
        x_index = find(arr == x_value);
    end
end