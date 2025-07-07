clear; clc; close all; tic;

% Load data
data = readtable("C:\Users\Aditya P J\Documents\MATLAB\Proyek 1 Matkom\master.csv");

% Preprocessing data
data.Properties.VariableNames = {'country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_per100k', ...
                                 'country_year', 'HDI_for_year', 'gdp_for_year', 'GDP_per_capita', 'generation'};

% Remove commas from 'gdp_for_year' column and convert to numeric
data.gdp_for_year = str2double(strrep(data.gdp_for_year, ',', ''));

% Fix 'age' column to handle '5-14 years'
data.age = strrep(data.age, '5-14 years', '05-14 years');

% List of countries to analyze
selected_countries = {'Indonesia', 'Japan', 'Korea', 'Malaysia', 'United Kingdom', 'Germany', ...
                      'Egypt', 'Morocco', 'United States', 'Brazil', 'Australia', 'France', 'Canada'};

% Initialize results
forecast_results = table();
evaluation_results = table();

% Create figure for plotting
figure;
hold on;

% Loop through each country
for i = 1:length(selected_countries)
    % Filter data by country
    country_data = data(strcmp(data.country, selected_countries{i}), :);

    % Aggregate suicides by year for the current country
    byyear = varfun(@sum, country_data, 'InputVariables', 'suicides_no', 'GroupingVariables', 'year');

    % Check if there are enough observations (minimum 10 data points)
    if height(byyear) < 10
        warning(['Skipping country: ' selected_countries{i} ' (insufficient data)']);
        continue;
    end

    % Prepare data
    X = byyear.year';               % Input: years
    T = byyear.sum_suicides_no';    % Target: suicides_no

    % Split data into training and testing (80% training, 20% testing)
    split_index = round(0.8 * length(X));
    if split_index >= length(X) - 1
        warning(['Skipping country: ' selected_countries{i} ' (not enough test data)']);
        continue;
    end

    X_train = X(1:split_index);
    T_train = T(1:split_index);
    X_test = X(split_index+1:end);
    T_test = T(split_index+1:end);

    % Neural Network (Feedforward)
    h = 5;  % Number of hidden neurons (can adjust based on model complexity)
    net = feedforwardnet(h, 'trainlm');  % Create feedforward network

    % Train the network
    net = train(net, X_train, T_train);

    % Make predictions on both training and test sets
    Y_train = net(X_train);  % Predictions on training set
    Y_test = net(X_test);    % Predictions on test set

    % Calculate error and evaluation metrics
    train_error = T_train - Y_train;
    test_error = T_test - Y_test;
    MSE_train = mse(train_error);
    MSE_test = mse(test_error);
    RMSE_test = sqrt(MSE_test);

    % Store evaluation results
    temp_eval = table(selected_countries(i), MSE_test, RMSE_test, ...
                      'VariableNames', {'Country', 'MSE', 'RMSE'});
    evaluation_results = [evaluation_results; temp_eval];

    % Future forecasting: predict for the next 10 years
    future_years = (X(end) + 1):(X(end) + 10);
    Y_future = net(future_years);

    % Store the forecasted results
    country_cell = repmat({selected_countries{i}}, length(future_years), 1);
    temp_results = table(country_cell, future_years', Y_future', ...
                         'VariableNames', {'Country', 'Year', 'Forecasted_Suicides'});
    forecast_results = [forecast_results; temp_results];

    % Plot actual vs predicted for test set and future forecast
    subplot(ceil(length(selected_countries)/3), 3, i);  % Create subplots for each country
    hold on;
    plot(X_train, T_train, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Training Data');
    plot(X_test, T_test, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Test Data');
    plot(X_test, Y_test, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Test');
    plot(future_years, Y_future, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Future Forecast');
    
    % Add title and labels
    title(['Forecast for ' selected_countries{i}]);
    xlabel('Year');
    ylabel('Number of Suicides');
    legend('show');
    hold off;
end

% Finalize plot
sgtitle('Neural Network Forecast and Model Evaluation for Selected Countries');
hold off;

% Handling NaN values in evaluation_results (using Mean Imputation)
nan_indices = isnan(evaluation_results.MSE);
mean_mse = mean(evaluation_results.MSE(~nan_indices));
evaluation_results.MSE(nan_indices) = mean_mse;

% Display results
disp('Forecast Results:');
disp(forecast_results);
disp('Evaluation Results (MSE and RMSE):');
disp(evaluation_results);

% Save forecast results to a CSV file
writetable(forecast_results, 'forecast_results_10_years_selected_countries_NN.csv');
writetable(evaluation_results, 'evaluation_results_selected_countries_NN.csv');