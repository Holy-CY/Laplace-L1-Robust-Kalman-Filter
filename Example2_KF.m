clear;
close all;
clc;

time = 0;
endtime = 60; % [sec]
dt = 0.1; % [sec]

nSteps = ceil((endtime - time)/dt);
 
result.time  = [];
result.xTrue = [];
result.xEst  = [];
result.PEst  = [];
result.Error = [];

% State Vector [x y yaw v]'
xEst = [0 10 0]';
 
% True State
xTrue = xEst;
 
% Covariance Matrix for motion
Q = diag([0.01 0.01 0.01]);
 
% Covariance Matrix for observation
R = 0.01;
R_outlier = 50 * R;

PEst = eye(3);

% system matrix
F = [1 dt dt^2;
     0 1 dt;
     0 0 1];
 
H = [0 1 0];
 
% L1 Robust Thuing Parameter
alpha = 0.1;
Error = [0 0 0]';

tic;
for i=1 : nSteps
    time = time + dt;
    
    % Observation True Value
    xTrue = F * xTrue + Q * randn(3, 1);
    z = H * xTrue + (1 - alpha) * R * randn(1, 1) + alpha * R_outlier * randn(1, 1);
    
    % ------ Kalman Filter --------
    % Predict
    xPred = F * xEst;
    PPred = F * PEst * F' + Q;
    
    % Update
    K    = (PPred * H') / (H * PPred * H' + R);
    xEst = xPred + K * (z - H * xPred);
    PEst = (eye(size(xEst,1)) - K * H) * PPred;
    
    Error(1) = sqrt(mean(xTrue(1) - xEst(1))^2);
    Error(2) = sqrt(mean(xTrue(2) - xEst(2))^2);
    Error(3) = sqrt(mean(xTrue(3) - xEst(3))^2);
    
    % Simulation Result
    result.time  = [result.time; time];
    result.xTrue = [result.xTrue; xTrue'];
    result.xEst  = [result.xEst;xEst'];
    result.PEst  = [result.PEst; diag(PEst)'];
    result.Error = [result.Error; Error'];
end
toc

DrawGraph(result);

function []=DrawGraph(result)
figure(1);
x=[ result.xTrue(:,1:3) result.xEst(:,1:3)];
set(gca, 'fontsize', 16, 'fontname', 'times');
plot(result.time, x(:,1), 'b', result.time, x(:,4), 'r');
grid on;
axis equal;

figure(2);
subplot(3, 1, 1);
plot(result.time, result.Error(:, 1), 'b');
xlim([0 60])
ylim([-0.05 0.6])
grid on;

subplot(3, 1, 2);
plot(result.time, result.Error(:, 2), 'b');
xlim([0 60])
ylim([-0.05 0.5])
grid on;

subplot(3, 1, 3);
plot(result.time, result.Error(:, 3), 'b');
xlim([0 60])
ylim([-0.05 0.5])
grid on;

end
   