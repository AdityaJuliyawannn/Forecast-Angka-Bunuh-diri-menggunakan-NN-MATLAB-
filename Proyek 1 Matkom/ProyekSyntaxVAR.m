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

% Loop through each country
for i = 1:length(selected_countries)
    % Filter data by country
    country_data = data(strcmp(data.country, selected_countries{i}), :);

    % Aggregate suicides by year and other variables for the current country
    byyear = varfun(@sum, country_data, 'InputVariables', {'suicides_no', 'population', 'GDP_per_capita'}, 'GroupingVariables', 'year');

    % Check if there are enough observations (minimum 10 data points)
    if height(byyear) < 10
        warning(['Skipping country: ' selected_countries{i} ' (insufficient data)']);
        continue;
    end

    % Prepare multivariate time series data
    X = [byyear.year, byyear.sum_suicides_no, byyear.sum_population, byyear.sum_GDP_per_capita];

    % Split data into training and testing (80% training, 20% testing)
    split_index = round(0.85 * size(X, 1));
    if split_index >= size(X, 1) - 1
        warning(['Skipping country: ' selected_countries{i} ' (not enough test data)']);
        continue;
    end

    X_train = X(1:split_index, :);
    X_test = X(split_index+1:end, :);

    % Fit VAR model using training data
    % Assuming 2 lags for simplicity, can be adjusted based on criteria like AIC/BIC
    model = varm(size(X_train, 2) - 1, 2); % Exclude year from number of variables
    EstModel = estimate(model, X_train(:, 2:end)); % Only use variables, exclude 'year'

    % Forecasting next 10 years
    forecast_horizon = 10;
    future_years = (X(end, 1) + 1):(X(end, 1) + forecast_horizon);
    Y_forecast = forecast(EstModel, forecast_horizon, X_train(:, 2:end));

    % Append forecasted years to the results
    country_cell = repmat({selected_countries{i}}, forecast_horizon, 1);
    temp_results = table(country_cell, future_years', Y_forecast(:, 1), ... % Suicides forecast
                         'VariableNames', {'Country', 'Year', 'Forecasted_Suicides'});
    forecast_results = [forecast_results; temp_results];

    % Evaluate model on test set
    Y_test_forecast = forecast(EstModel, size(X_test, 1), X_train(:, 2:end));

    % Calculate evaluation metrics (MSE, RMSE)
    test_error = X_test(:, 2) - Y_test_forecast(:, 1); % Comparing actual suicides with forecasted
    MSE_test = mean(test_error.^2);
    RMSE_test = sqrt(MSE_test);

    % Store evaluation results
    temp_eval = table(selected_countries(i), MSE_test, RMSE_test, ...
                      'VariableNames', {'Country', 'MSE', 'RMSE'});
    evaluation_results = [evaluation_results; temp_eval];

    % Plot actual vs predicted for test set and future forecast
    figure;
    hold on;
    plot(X(1:split_index, 1), X_train(:, 2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Training Data');
    plot(X(split_index+1:end, 1), X_test(:, 2), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Test Data');
    plot(X(split_index+1:end, 1), Y_test_forecast(:, 1), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Test');
    plot(future_years, Y_forecast(:, 1), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Future Forecast');
    
    % Add title and labels
    title(['Forecast for ' selected_countries{i}]);
    xlabel('Year');
    ylabel('Number of Suicides');
    grid on;
    legend('show');
    hold off;
end

% Finalize plot
sgtitle('VAR Model Forecast and Evaluation for Selected Countries');
hold off;

% Display results
disp('Forecast Results:');
disp(forecast_results);
disp('Evaluation Results (MSE and RMSE):');
disp(evaluation_results);

% Save forecast results to a CSV file
writetable(forecast_results, 'forecast_results_10_years_selected_countries_VAR.csv');
writetable(evaluation_results, 'evaluation_results_selected_countries_VAR.csv');
