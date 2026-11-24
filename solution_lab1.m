% =========================================================================
% IS-Lab 1: Classification using Single Perceptron & Naive Bayes
% =========================================================================

clear all;
close all;
clc;

%% 1. READ IMAGES
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

%% 2. FEATURE EXTRACTION (Color & Roundness)
% For Apples
hsv_value_A1=spalva_color(A1); metric_A1=apvalumas_roundness(A1);
hsv_value_A2=spalva_color(A2); metric_A2=apvalumas_roundness(A2);
hsv_value_A3=spalva_color(A3); metric_A3=apvalumas_roundness(A3);
hsv_value_A4=spalva_color(A4); metric_A4=apvalumas_roundness(A4);
hsv_value_A5=spalva_color(A5); metric_A5=apvalumas_roundness(A5);
hsv_value_A6=spalva_color(A6); metric_A6=apvalumas_roundness(A6);
hsv_value_A7=spalva_color(A7); metric_A7=apvalumas_roundness(A7);
hsv_value_A8=spalva_color(A8); metric_A8=apvalumas_roundness(A8);
hsv_value_A9=spalva_color(A9); metric_A9=apvalumas_roundness(A9);

% For Pears
hsv_value_P1=spalva_color(P1); metric_P1=apvalumas_roundness(P1);
hsv_value_P2=spalva_color(P2); metric_P2=apvalumas_roundness(P2);
hsv_value_P3=spalva_color(P3); metric_P3=apvalumas_roundness(P3);
hsv_value_P4=spalva_color(P4); metric_P4=apvalumas_roundness(P4);

%% 3. PREPARE TRAINING AND TESTING DATA
% Training set: 3 Apples (A1, A2, A3) and 2 Pears (P1, P2) -> 5 samples
x1_train = [hsv_value_A1, hsv_value_A2, hsv_value_A3, hsv_value_P1, hsv_value_P2];
x2_train = [metric_A1,    metric_A2,    metric_A3,    metric_P1,    metric_P2];
P_train = [x1_train; x2_train];
T_train = [1; 1; 1; -1; -1]; % Apple = 1, Pear = -1

% Testing set: remaining 6 Apples and 2 Pears -> 8 test samples
x1_test = [hsv_value_A4, hsv_value_A5, hsv_value_A6, hsv_value_A7, hsv_value_A8, hsv_value_A9, hsv_value_P3, hsv_value_P4];
x2_test = [metric_A4,    metric_A5,    metric_A6,    metric_A7,    metric_A8,    metric_A9,    metric_P3,    metric_P4];
P_test = [x1_test; x2_test];
T_test = [1; 1; 1; 1; 1; 1; -1; -1];

%% 4. PERCEPTRON TRAINING (Main Task - 8 Points)
fprintf('========================================\n');
fprintf('  STARTING PERCEPTRON TRAINING\n');
fprintf('========================================\n');

% Random parameter initialization
w1 = randn(1);
w2 = randn(1);
b  = randn(1);
eta = 0.1; % Learning rate (0 < eta < 1)

epoch = 0;
total_error = 1;
N = length(T_train);

while total_error ~= 0
    epoch = epoch + 1;
    total_error = 0;
    
    for i = 1:N
        % 1. Forward Pass: Calculate weighted sum
        v = x1_train(i)*w1 + x2_train(i)*w2 + b;
        
        % 2. Activation Function (Signum / Threshold)
        if v > 0
            y = 1;
        else
            y = -1;
        end
        
        % 3. Calculate Error
        e = T_train(i) - y;
        
        % 4. Parameter Update (Delta Rule)
        w1 = w1 + eta * e * x1_train(i);
        w2 = w2 + eta * e * x2_train(i);
        b  = b  + eta * e;
        
        total_error = total_error + abs(e);
    end
    
    fprintf('Epoch %d: Total Error = %d\n', epoch, total_error);
    if epoch > 5000
        fprintf('Safety stop reached (5000 epochs).\n');
        break;
    end
end

fprintf('\nTraining completed in %d epochs!\n', epoch);
fprintf('Trained parameters:\n  w1 = %.4f\n  w2 = %.4f\n  b  = %.4f\n\n', w1, w2, b);

%% 5. TEST PERCEPTRON ON UNSEEN TEST DATA
fprintf('========================================\n');
fprintf('  PERCEPTRON TEST RESULTS (8 Unseen Images)\n');
fprintf('========================================\n');
test_N = length(T_test);
perceptron_predictions = zeros(test_N, 1);
correct_perceptron = 0;

for j = 1:test_N
    v = P_test(1, j)*w1 + P_test(2, j)*w2 + b;
    if v > 0
        y = 1;
    else
        y = -1;
    end
    perceptron_predictions(j) = y;
    if y == T_test(j)
        correct_perceptron = correct_perceptron + 1;
    end
    fprintf('Test Image %d -> Predicted: %2d, True: %2d\n', j, y, T_test(j));
end
fprintf('Perceptron Test Accuracy: %.2f%%\n\n', (correct_perceptron / test_N) * 100);

%% 6. NAIVE BAYES CLASSIFIER (Additional Task - +2 Points)
fprintf('========================================\n');
fprintf('  NAIVE BAYES CLASSIFIER (Self-Contained)\n');
fprintf('========================================\n');

% Train Naive Bayes using Gaussian likelihood
% Separate training data by class
idx_apple = find(T_train == 1);
idx_pear  = find(T_train == -1);

% Priors: P(Class)
prior_apple = length(idx_apple) / N;
prior_pear  = length(idx_pear) / N;

% Class Apple: Mean and Variance for feature 1 and feature 2
mu_apple_x1 = mean(x1_train(idx_apple));
var_apple_x1 = var(x1_train(idx_apple));
mu_apple_x2 = mean(x2_train(idx_apple));
var_apple_x2 = var(x2_train(idx_apple));

% Class Pear: Mean and Variance for feature 1 and feature 2
mu_pear_x1 = mean(x1_train(idx_pear));
var_pear_x1 = var(x1_train(idx_pear));
mu_pear_x2 = mean(x2_train(idx_pear));
var_pear_x2 = var(x2_train(idx_pear));

% Gaussian PDF lambda function
gauss_pdf = @(x, mu, v) (1 / sqrt(2*pi*v)) * exp(-((x - mu).^2) / (2*v));

% Predict on test data
nb_predictions = zeros(test_N, 1);
correct_nb = 0;

for j = 1:test_N
    x1_val = P_test(1, j);
    x2_val = P_test(2, j);
    
    % Likelihood * Prior for Apple
    p_x_apple = gauss_pdf(x1_val, mu_apple_x1, var_apple_x1) * ...
                gauss_pdf(x2_val, mu_apple_x2, var_apple_x2) * prior_apple;
            
    % Likelihood * Prior for Pear
    p_x_pear  = gauss_pdf(x1_val, mu_pear_x1, var_pear_x1) * ...
                gauss_pdf(x2_val, mu_pear_x2, var_pear_x2) * prior_pear;
            
    if p_x_apple >= p_x_pear
        y_nb = 1;
    else
        y_nb = -1;
    end
    
    nb_predictions(j) = y_nb;
    if y_nb == T_test(j)
        correct_nb = correct_nb + 1;
    end
    fprintf('Test Image %d -> NB Predicted: %2d, True: %2d (P(A)=%.2e, P(P)=%.2e)\n', ...
        j, y_nb, T_test(j), p_x_apple, p_x_pear);
end
fprintf('Naive Bayes Test Accuracy: %.2f%%\n', (correct_nb / test_N) * 100);
