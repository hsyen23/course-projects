function myArray = expand_zigzag(matrix)
    load("Zig-Zag Pattern.txt");
    myArray = zeros(1, 64);
    for row = 1 : size(matrix,1)
        for column = 1 : size(matrix,2)
            number = Zig_Zag_Pattern(row, column) + 1;
            myArray(number) = matrix(row, column);
        end
    end
end