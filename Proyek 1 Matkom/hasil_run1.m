clear; clc; close all; tic
addpath(genpath(pwd));

%% FORECASTING SUICIDE WITH SIMPLE AR (AUTOREGRESSIVE MODEL)
% Author : Aditya Pratama Juliyawan, Lavina Mutia Dewi, Asika Duri
% Kelompok 6 MATKOM

% Read Data
data = readtable("Proyek 1 Matkom\master.csv");
dates = datetime(data.year, 1, 1);

% Set target prediksi
suicides = data.suicides_no;

% Split data in-sample (training), hyper-parameter tuning (validation), and
% out-of-sample (test) -- (1/3, 1/3, 1/3) split
len = ceil(length(dates) / 3);
train_ind = 1:len; 
val_ind   = (len+1):(2*len);
test_ind  = (2*len+1):length(dates);

% Target prediksi suicides
YY = suicides; 
% Pick forecasting horizon
h = 3; 

% Pick maximum number of lags
p_max = 15;
% Pilih in-sample information criterion untuk model selection
ICind = 2; 

% Rolling window indicator untuk pseudo-oos (default: expanding)
roll = 0; 

% Rolling window length (jika roll=1)
wL = 36;

% Tentukan benchmark untuk out-of-sample performance comparison
benchmark = 'RW';

% Max lag order untuk predictor
py_max = 15;

% Index semua kombinasi hyperparameter
combinations = reshape(ndgrid(1:py_max),[],1);

% Preallocate untuk validation (lag-length) selection 
insampIC = NaN(size(combinations,1),3);

% Implement autoregressive model with in-sample lag length selection
for i = 1:size(combinations,1)
    p = combinations(i,1);
    
    % Inisialisasi target dan predictors
    Y = YY(train_ind);
    X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); X = [ones(length(X),1), X];
    Y = Y((1+h+p):end);
    
    % Estimasi OLS
    S.Beta = regress(Y, X);
    
    % Evaluasi pada validation set
    Y = YY(val_ind);
    X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); Xval = [ones(length(X),1), X];
    Yval = Y((1+h+p):end);
    yhat = Xval * S.Beta;
    
    % Simpan hasil
    insampIC(p,:) = IC(Yval, yhat, length(Yval), size(Xval,2)); 
end

% Pilih model terbaik
[~,best_i] = min(insampIC(:,ICind)); 
p = combinations(best_i,1);

% Preallocate out-of-sample forecast
yHat = NaN(length(test_ind),1);   

if roll
    for t = 1:length(test_ind)
        % Inisialisasi data hingga t pada test set
        endInd = test_ind(t) - h - 1;

        Y = YY((endInd-wL-p-h):endInd);
        X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); X = [ones(length(X),1), X];
        Y = Y((1+h+p):end);
        
        % Estimasi OLS
        S = OLS(X,Y);
        
        % Prediksi untuk t+1
        Y = YY((endInd-p+1):(endInd + 1));
        X = [1, Y'];
        
        yHat(t) = X * S.Beta;
    end
else
    for t = 1:length(test_ind)
        % Inisialisasi data hingga t pada test set
        endInd = test_ind(t) - h- 1;

        Y = YY(1:endInd);
        X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); X = [ones(length(X),1), X];
        Y = Y((1+h+p):end);
        
        % Estimasi OLS
        S = OLS(X,Y);
        
        % Prediksi untuk t+1
        Y = YY((endInd-p+1):(endInd + 1));
        X = [1, Y'];
        
        yHat(t) = X * S.Beta;
    end
end

% Nilai aktual
Y = YY(test_ind);

% Forecast from AR model
yHat = yHat(:); % Ensure yHat is a column vector
e1 = Y - yHat;

% Forecast from benchmark
ytemp = forecast_RW(YY);
ytemp = ytemp(test_ind); % Ensure ytemp is aligned with test_ind
e2 = Y - ytemp;

% Pastikan Y dan yHat tidak mengandung NaN
valid_idx = ~isnan(Y) & ~isnan(yHat);
Y_clean = Y(valid_idx);
yHat_clean = yHat(valid_idx);

% Hitung MSPE hanya dengan data yang valid
eval = MSPE(Y_clean, yHat_clean, wL);

% Tangani nilai NaN pada hasil MSPE
eval.CUM_MSPE(isnan(eval.CUM_MSPE)) = 0;
eval.ROLL_RMSPE(isnan(eval.ROLL_RMSPE)) = nanmean(eval.ROLL_RMSPE);

% Tampilkan hasil MSPE
disp(['Out-of-sample total MSPE is: ', num2str(eval.MSPE)]);

% Memanggil forecast berdasarkan benchmark
switch benchmark
    case 'RW'
        ytemp = forecast_RW(YY);  % Memanggil fungsi forecast_RW
    case 'PM'
        ytemp = forecast_PM(YY);  % Memanggil fungsi forecast_PM
    otherwise
        error('Benchmark tidak dikenal. Pilih RW atau PM.');
end

% Periksa apakah ukuran YY(test_ind) dan ytemp(test_ind) sama
if length(YY(test_ind)) ~= length(ytemp(test_ind))
    error('Mismatch in the size of actual and forecast data. Check forecast functions.');
end

% Evaluasi benchmark
bench_eval = MSPE(YY(test_ind), ytemp(test_ind), wL);

% Periksa dan tangani nilai NaN pada MSPE benchmark
bench_eval.CUM_MSPE(isnan(bench_eval.CUM_MSPE)) = 0;
bench_eval.ROLL_RMSPE(isnan(bench_eval.ROLL_RMSPE)) = nanmean(bench_eval.ROLL_RMSPE);

% Samakan panjang data untuk plotting
min_length = min(length(eval.CUM_MSPE), length(bench_eval.CUM_MSPE));
dates_plot = dates(test_ind(1:min_length));
CUM_MSPE_plot = eval.CUM_MSPE(1:min_length);
bench_CUM_MSPE_plot = bench_eval.CUM_MSPE(1:min_length);

% Plot Cumulative MSPE
plot_ts(dates_plot, [CUM_MSPE_plot, bench_CUM_MSPE_plot], '', 'Cumulative RMSE', 2, {'AR', benchmark}, 0);

% Rolling RMSPE
min_length = min(length(eval.ROLL_RMSPE), length(bench_eval.ROLL_RMSPE));
dates_plot = dates(test_ind((1+wL):(1+wL+min_length-1)));
ROLL_RMSPE_plot = eval.ROLL_RMSPE(1:min_length);
bench_RMSPE_plot = bench_eval.ROLL_RMSPE(1:min_length);

% Plot Rolling RMSPE
plot_ts(dates_plot, [ROLL_RMSPE_plot, bench_RMSPE_plot], '', 'Rolling RMSE', 3, {'AR', benchmark}, 0);

% OOS PERFORMANCE TESTS
e1 = eval.errors;
e2 = bench_eval.errors;

% Ensure e1 and e2 are of the same length before testing
if length(e1) ~= length(e2)
    error('e1 and e2 must be of the same length for DM test.');
end

[DM, pval_L, pval_LR, pval_R] = dmtest(e1, e2, h);

% Example using bootstrapCI for bootstrapped confidence intervals
[mean_bootstrap, CI_low, CI_high] = bootstrapCI(e2 - e1, 1000, 0.05);

% Model confidence set
[includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ] = mcs([e1, e2], 0.05, 1000, 12, 'STATIONARY');

% Save errors
AR_errors = e1;
if isfile('FORECAST_ERRORS.mat'), save('FORECAST_ERRORS.mat', 'AR_errors','-append'); end

%% FUNCTION: IC
function [IC] = IC(Y, yhat, n, k)
    % Calculate the Information Criteria: AIC, BIC, and HQ
    % Inputs:
    %   Y - actual values
    %   yhat - predicted values
    %   n - number of observations
    %   k - number of parameters (including the intercept)

    % Residual sum of squares
    RSS = sum((Y - yhat).^2);
    
    % AIC, BIC, and HQ
    AIC = n * log(RSS/n) + 2 * k;
    BIC = n * log(RSS/n) + k * log(n);
    HQ = n * log(RSS/n) + 2 * k * log(log(n));
    
    % Output IC as a row vector
    IC = [AIC, BIC, HQ];
end

%% FUNCTION: OLS
function S = OLS(X, Y)
    % Ordinary Least Squares (OLS) Estimation
    % Inputs:
    %   X - matrix of predictors
    %   Y - vector of response variable

    % Estimate OLS
    S.Beta = regress(Y, X);
end

%% FUNCTION: forecast_RW
function yhat = forecast_RW(Y)
    % Forecast using Random Walk method
    % Input:
    %   Y - vector of historical data
    % Output:
    %   yhat - vector of forecasts using the Random Walk method

    lastValue = Y(end);
    yhat = repmat(lastValue, size(Y));
end

%% FUNCTION: MSPE
function eval = MSPE(actual, forecast, windowLength)
    % MSPE: Calculates the Mean Squared Prediction Error and related metrics
    % actual: The actual observed values
    % forecast: The forecasted values
    % windowLength: The length of the rolling window for RMSPE

    errors = (actual - forecast).^2;
    MSPE_value = mean(errors);
    CUM_MSPE_value = cumsum(errors) ./ (1:length(errors))';
    ROLL_RMSPE_value = nan(length(errors), 1);
    
    for i = windowLength:length(errors)
        ROLL_RMSPE_value(i) = sqrt(mean(errors((i-windowLength+1):i)));
    end
    
    eval.MSPE = MSPE_value;
    eval.CUM_MSPE = CUM_MSPE_value;
    eval.ROLL_RMSPE = ROLL_RMSPE_value;
    eval.errors = errors;
end

%% FUNCTION: plot_ts
function plot_ts(x, y, title_str, ylabel_str, linewidth, legend_labels, grid_on)
    % Plot time series data with custom formatting
    % x: The x-axis data (usually dates)
    % y: The y-axis data (matrix with columns of different series)
    % title_str: Title of the plot
    % ylabel_str: Label for the y-axis
    % linewidth: Line width for the plot lines
    % legend_labels: Cell array of strings for the legend
    % grid_on: 0 or 1 to turn grid on or off

    plot(x, y, 'LineWidth', linewidth);
    title(title_str);
    xlabel('Date');
    ylabel(ylabel_str);
    
    if ~isempty(legend_labels)
        legend(legend_labels, 'Location', 'best');
    end
    
    if grid_on
        grid on;
    else
        grid off;
    end
end

function [DM, pval_L, pval_LR, pval_R] = dmtest(e1, e2, h)
    % DM test to compare forecast accuracy
    % e1, e2: Forecast errors
    % h: Forecast horizon

    % Ensure e1 and e2 are the same length
    if length(e1) ~= length(e2)
        error('Error vectors e1 and e2 must be the same length.');
    end
    
    % Compute the DM statistic
    d = e1 - e2;
    d_bar = mean(d);
    var_d = var(d);

    % Calculate the DM test statistic
    DM = d_bar / sqrt(var_d / length(d));

    % Compute p-values for various tests (dummy values here)
    pval_L = 1 - normcdf(DM);
    pval_LR = 1 - normcdf(DM); % Placeholder, adjust according to actual test
    pval_R = 1 - normcdf(DM); % Placeholder, adjust according to actual test
end

% Placeholder implementation for MZ test
function [MZstat, MZpval] = MZtest(Y, yHat)
    % Compute the forecast errors
    errors = Y - yHat;

    % Placeholder for MZ statistic calculation
    MZstat = mean(errors) / std(errors); % Replace with actual computation

    % Placeholder p-value calculation
    MZpval = 1 - normcdf(MZstat); % Replace with actual p-value computation
end

function [c, u, l] = bsds(e2, e1, numBootstrap, param1, param2)
    % Basic placeholder implementation of a bootstrapping function for standard deviation
    % e2, e1: Error vectors
    % numBootstrap: Number of bootstrap iterations
    % param1, param2: Additional parameters

    % Placeholder implementation, actual function will vary
    bootstraps = bootstrp(numBootstrap, @std, e2 - e1);

    % Calculate percentile-based confidence intervals
    c = mean(bootstraps); % Placeholder
    u = prctile(bootstraps, 97.5); % Upper bound for 95% CI
    l = prctile(bootstraps, 2.5);  % Lower bound for 95% CI
end