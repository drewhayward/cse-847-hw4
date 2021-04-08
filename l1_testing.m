par = [0.00000001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

% Load the data
load ad_data/ad_data.mat;
[N_train, d] = size(X_train);
[N_test, d] = size(X_test);

y_test_01 = y_test;
y_train_01 = y_train;
y_test(y_test==0) = -1;
y_train(y_train==0) = -1;

aucs = [];
for i = 1:length(par)
    % Train model
    [w, c] = logistic_l1_train(X_train, y_train, par(i));

    probs = 1 ./ (1+ exp(- (X_test * w + c)));

    [X,Y,T, AUC] = perfcurve(y_test_01, probs, 1);
    aucs(end+1) = AUC;
end


plot(par, aucs);
ylabel('Classification AUC')
xlabel("L1 Reg Term")