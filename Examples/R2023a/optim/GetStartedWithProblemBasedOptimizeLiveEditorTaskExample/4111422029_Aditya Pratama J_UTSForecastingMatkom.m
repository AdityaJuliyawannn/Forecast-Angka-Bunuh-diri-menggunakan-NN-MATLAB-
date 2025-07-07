clear; clc; close all; tic

X = readmatrix('C:\Users\Aditya P J\Documents\Matematika\Semester 5\Matkom\Data_Historis_JakartaStockExchangeComposite.xlsx', 'Sheet', 'data', 'Range', 'A2:E151')';
T = readmatrix('C:\Users\Aditya P J\Documents\Matematika\Semester 5\Matkom\Data_Historis_JakartaStockExchangeComposite.xlsx', 'Sheet', 'data', 'Range', 'F2:F151')';
t = 1:1:length(T);

h = 5;
net = feedforwardnet (h, 'trainlm');
net = train(net,X,T);
Y = sim(net,X);
e = T-Y;
RMSE = sqrt(mse(e));

Xt = readmatrix('C:\Users\Aditya P J\Documents\MATLAB\Trying 1\ANTM.xlsx', 'Sheet', 'data', 'Range', 'A151:E151')';
Yt = sim(net,Xt);

tab = [t' T' Y' e'];
h
Yt
RMSE

plot(t,T,'-',t,Y,'-','LineWidth',1.2);
title('Peramalan Data Saham PT Antam Tbk');
xlabel('Time(t)');ylabel('Harga Saham Penutupan');
legend('Aktual Harga Saham ANTM','Prediksi Harga Saham ANTM','Orientation','horizontal','Location','bestoutside');