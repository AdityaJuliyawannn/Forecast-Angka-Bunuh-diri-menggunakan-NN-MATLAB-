clear; clc; close all; tic;

%% FORECASTING SUICIDE WITH SIMPLE AR (NEURAL NETWORK)
% Author : Aditya Pratama Juliyawan, Lavina Mutia Dewi, Asika Duri
% Kelompok 6 MATKOM

% Import data
data = readtable("C:\Users\Aditya P J\Documents\MATLAB\Proyek 1 Matkom\master.csv"); 

%% Preprocessing data
data.Properties.VariableNames = {'country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_per100k', ...
                                 'country_year', 'HDI_for_year', 'gdp_for_year', 'GDP_per_capita', 'generation'};
data.gdp_for_year = str2double(strrep(data.gdp_for_year, ',', '')); % Menghilangkan Koma
data.age = strrep(data.age, '5-14 years', '05-14 years'); % Membenarkan format umur

% Memilih negara yang akan di prediksi
selected_countries = {'Germany', 'Japan', 'Brazil', 'United States', 'Australia'}; % Some countries

% Inisiasi Penyimpanan Hasil
forecast_results = table();
evaluation_results = table();

% Penanganan Error dan Perbaikan Model
for i = 1:length(selected_countries)
    % Filter data untuk negara sekarang 
    country_data = data(strcmp(data.country, selected_countries{i}), :);
    
    % Aggregate bunnuh diri dari tahun
    byyear = varfun(@sum, country_data, 'InputVariables', 'suicides_no', 'GroupingVariables', 'year');
    
    % Periksa data cukp (at least 10 data)
    if height(byyear) < 10
        warning(['Skipping country: ' selected_countries{i} ' (insufficient data)']);
        continue;
    end
    
    % split data training 80% dan test 20%
    split_index = round(0.8 * height(byyear));
    train_data = byyear.sum_suicides_no(1:split_index);
    test_data = byyear.sum_suicides_no(split_index+1:end);
    train_years = byyear.year(1:split_index);
    test_years = byyear.year(split_index+1:end);
    
    % ARIMA model fitting
    model = arima('Constant', 0, 'D', 1, 'Seasonality', 0);
    try
        estimated_model = estimate(model, train_data);
    catch ME
        disp(['Error for country: ' selected_countries{i} ' - ' ME.message]);
        warning(['ARIMA model failed for country: ' selected_countries{i}]);
        continue;
    end
    
    % Forecasting on test data
    try
        test_forecast = forecast(estimated_model, length(test_data), 'Y0', train_data);
        MSE_value = mean((test_forecast - test_data).^2);
        RMSE_value = sqrt(MSE_value);
    catch ME
        disp(['Forecasting error for country: ' selected_countries{i} ' - ' ME.message]);
        warning(['Forecasting failed for country: ' selected_countries{i}]);
        MSE_value = NaN;
        RMSE_value = NaN;
    end
    
    % Store evaluation results
    eval_temp = table(selected_countries(i), MSE_value, RMSE_value, 'VariableNames', {'Country', 'MSE', 'RMSE'});
    evaluation_results = [evaluation_results; eval_temp];

    % Future forecasting (next 10 years)
    forecast_years = 10;
    try
        future_forecast = forecast(estimated_model, forecast_years, 'Y0', byyear.sum_suicides_no);
    catch ME
        disp(['Future forecasting error for country: ' selected_countries{i} ' - ' ME.message]);
        warning(['Future forecasting failed for country: ' selected_countries{i}]);
        future_forecast = NaN(forecast_years, 1);
    end
    forecast_time = (byyear.year(end)+1):(byyear.year(end)+forecast_years);
    
    % Store forecast results
    country_cell = repmat({selected_countries{i}}, length(forecast_time), 1);
    result_temp = table(country_cell, forecast_time', future_forecast, ...
                        'VariableNames', {'Country', 'Year', 'Forecasted_Suicides'});
    forecast_results = [forecast_results; result_temp];
    
    % Plot actual and forecasted data
    figure;
    hold on;
    plot(train_years, train_data, 'b-', 'LineWidth', 1.5);
    plot(test_years, test_data, 'r-', 'LineWidth', 1.5);
    plot(test_years, test_forecast, 'g--', 'LineWidth', 1.5);
    plot(forecast_time, future_forecast, 'k--', 'LineWidth', 1.5);
    title(['Suicide Forecast for ' selected_countries{i}]);
    xlabel('Year');
    ylabel('Number of Suicides');
    legend('Training Data', 'Testing Data', 'Test Forecast', 'Future Forecast');
    hold off;
end

% Handle NaN in evaluation results using Mean Imputation
nan_indices = isnan(evaluation_results.MSE);
mean_mse = mean(evaluation_results.MSE(~nan_indices), 'omitnan');
evaluation_results.MSE(nan_indices) = mean_mse;

% Handle NaN in RMSE results similarly
nan_indices_rmse = isnan(evaluation_results.RMSE);
mean_rmse = mean(evaluation_results.RMSE(~nan_indices_rmse), 'omitnan');
evaluation_results.RMSE(nan_indices_rmse) = mean_rmse;

% Display results
disp('Forecast Results:');
disp(forecast_results);
disp('Evaluation Results (MSE and RMSE):');
disp(evaluation_results);

% Save results to CSV files
writetable(forecast_results, 'forecast_results_selected_countries.csv');
writetable(evaluation_results, 'evaluation_results_selected_countries.csv');
