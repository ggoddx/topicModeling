% This script test the function gibbs.m
clear 
close all
% Generate a document
Doc = {[1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6] + 1;
       [2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2] + 1;
       [1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0] + 1;
       [5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0] + 1;
       [2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0] + 1;
       [5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2] + 1;};
% Vocabulary
W = 7;
D = length(Doc);
% Topic 
T = 2;
% Choose alpha and beta
alpha = 2;
beta = .5;
% set up parameter
Iter = 10000;
BURN_IN = 2000;
Sampler_lag = 10;
% Start Algorithm
disp('LDA using Gibbs Sampling.');
[theta, phi, avetheta, avephi] = gibbs(alpha, beta, Doc, T, W, Iter, BURN_IN, Sampler_lag)


