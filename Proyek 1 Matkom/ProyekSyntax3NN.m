clear; clc; close all; tic;

%% FORECASTING SUICIDE WITH SIMPLE AR (NEURAL NETWORK)
% Author : Aditya Pratama Juliyawan, Lavina Mutia Dewi, Asika Duri
% Kelompok 6 MATKOM

% Muat data
data = readtable("C:\Users\Aditya P J\Documents\MATLAB\Proyek 1 Matkom\master.csv");

%% Preprocessing data
data.Properties.VariableNames = {'country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_per100k', ...
                                 'country_year', 'HDI_for_year', 'gdp_for_year', 'GDP_per_capita', 'generation'};

% Remove commas from 'gdp_for_year' column and convert to numeric
data.gdp_for_year = str2double(strrep(data.gdp_for_year, ',', ''));

% Membenarkan kolom 'age' untuk menangani '5-14 years'
data.age = strrep(data.age, '5-14 years', '05-14 years');

% Memilih negara yang akan di prediksi
selected_countries = {'Singapore', 'Japan', 'Philippines','Kazakhstan', 'South Africa','United States', 'Puerto Rico', 'Brazil', 'Grenada', 'Austria', 'Russian Federation', 'Bulgaria', 'Ukraine'};

% Inisiasi Hasil 
forecast_results = table();
evaluation_results = table();

% Loop setiap negara
for i = 1:length(selected_countries)
    % Filter data berdasarkan negara
    country_data = data(strcmp(data.country, selected_countries{i}), :);

    % Jumlah total bunuh diri per tahun di negara saat ini
    byyear = varfun(@sum, country_data, 'InputVariables', 'suicides_no', 'GroupingVariables', 'year');

    % Periksa data cukp (at least 10 data)
    if height(byyear) < 10
        warning(['Skipping country: ' selected_countries{i} ' (insufficient data)']);
        continue;
    end

    % Prepare data
    X = byyear.year';               % Input: years
    T = byyear.sum_suicides_no';    % Target: suicides_no

    % Split data training dan testing (85% training, 15% testing)
    split_index = round(0.85 * length(X));
    if split_index >= length(X) - 1
        warning(['Skipping country: ' selected_countries{i} ' (not enough test data)']);
        continue;
    end

    X_train = X(1:split_index);
    T_train = T(1:split_index);
    X_test = X(split_index+1:end);
    T_test = T(split_index+1:end);

    % Normalisasi Data (input dan target)
    [X_train, X_train_settings] = mapminmax(X_train, 0, 1);
    [T_train, T_train_settings] = mapminmax(T_train, 0, 1);
    X_test = mapminmax('apply', X_test, X_train_settings);
    T_test = mapminmax('apply', T_test, T_train_settings);

    % Neural Network
    h_values = [10, 20, 30];  % Jumlah neuron tersembunyi
    best_rmse = inf;
    best_mse = inf;
    best_net = [];

    for h = h_values
        % Buat jaringan feedforward dengan regularisasi Bayesian
        net = feedforwardnet(h, 'trainbr');  % Regulasi Bayes
        net.trainParam.showWindow = true; 
        net.trainParam.epochs = 250;

        % Latih Jaringan
        net = train(net, X_train, T_train);

        % Membuat predikisi dari Training dan Testing
        Y_train = net(X_train);  
        Y_test = net(X_test);    
        
        % Menghitung Error dan Evaluasi Matriks
        test_error = T_test - Y_test;
        MSE_test = mse(test_error);  
        RMSE_test = sqrt(MSE_test);  

        % Lacak jaringan terbaik berdasarkan RMSE
        if RMSE_test < best_rmse
            best_rmse = RMSE_test;
            best_mse = MSE_test;
            best_net = net;
        end
    end

    % Buat prediksi dengan model terbaik
    Y_train_best = best_net(X_train);
    Y_test_best = best_net(X_test);

    % Denormalisasi prediksi
    Y_train_best = mapminmax('reverse', Y_train_best, T_train_settings);
    Y_test_best = mapminmax('reverse', Y_test_best, T_train_settings);
    T_train = mapminmax('reverse', T_train, T_train_settings);
    T_test = mapminmax('reverse', T_test, T_train_settings);

    % Simpan hasil evaluasi RMSE dan MSE
    temp_eval = table(selected_countries(i), best_mse, best_rmse, ...
                      'VariableNames', {'Country', 'MSE', 'RMSE'});
    evaluation_results = [evaluation_results; temp_eval];

    % Prediksi MasaDepan: Prediksi 10 tahun kedepan 
    future_years = (X(end) + 1):(X(end) + 10);
    future_years_normalized = mapminmax('apply', future_years, X_train_settings);
    Y_future = best_net(future_years_normalized);
    Y_future = mapminmax('reverse', Y_future, T_train_settings);  % Denormalize future predictions

    % Simpan hasil Forecasting
    country_cell = repmat({selected_countries{i}}, length(future_years), 1);
    temp_results = table(country_cell, future_years', Y_future', ...
                         'VariableNames', {'Country', 'Year', 'Forecasted_Suicides'});
    forecast_results = [forecast_results; temp_results];

    % Plot hanya hasil prediksi training dari awal periode
    figure;
    hold on;
    plot(X, T, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Actual Data');  % Plot actual data
    plot(X(1:split_index), Y_train_best, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Training');  % Predicted for training set
    plot(X(split_index+1:end), Y_test_best, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Test');  % Predicted for test set
    plot(future_years, Y_future, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Future Forecast');  % Future forecast

    % Tambahkan Judul
    title(['Neural Network Forecast for ' selected_countries{i}]);
    xlabel('Tahun');
    ylabel('Angka Bunuh Diri');
    grid on;
    legend('show');
    hold off;
end

% Finalisasi plot
sgtitle('Neural Network Forecast and Model Evaluation for Selected Countries');
hold off;

% Display hasil
disp('Forecast Results:');
disp(forecast_results);
disp('Evaluation Results (MSE and RMSE):');
disp(evaluation_results);

% Simpan forecast di File CSV
writetable(forecast_results, 'forecast_results_10_years_selected_countries_NN.csv');
writetable(evaluation_results, 'evaluation_results_selected_countries_NN.csv');
