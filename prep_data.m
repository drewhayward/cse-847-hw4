function out_data = prep_data(data)
    shape = size(data);
    out_data = [data, ones(shape(1), 1)];
end