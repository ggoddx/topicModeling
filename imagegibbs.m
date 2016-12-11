% This script generate predefined images, each representing a topic
% Then perform random sampling to form a corpus and test on gibbs.m

% Step 1. Load predefined topic
% @topic 10-by-1 cell, 5-by-5 matrix for each cell
close all
clear
gentopic;
%load 'predefinedtopic';

% Step 2. For each document(image) sample a multinomial distribution form Dirichlet Distribution with alpha = 1
% @param 
%	 D:= number of documnets 
% @param 
%	 alpha:= Dirichlet parameter, 1-by-T vector
% @param 
%	 T:= number of topics in each documents, here T = 10
% @return 
%	 theta:= document topic distribution, D-by-T matrix

T = length(topic);
alpha = ones(1,T);
D = 1000; % generate 1000 documents
theta = drchrnd(alpha, D); % sample theta from Dirichlet with smooth parameter alpha = 1

% Step 3. For each document, given the topic distribution, sample words the topic;
Doc = cell(D,1);
for d = 1:D
    doc = wordsampling(topic,theta(d,:));
    Doc{d,1} = doc;
end

% show 30 documents
for i = 1:30
    subplot(3,10,i);
    showdoc(Doc{i,1});
    title(['document',num2str(i)]);
end

% Step 4. Gibbs Sampling 
alpha = 1;
beta = 0.1;
T = 10; W = 25; 
Iter = 500; BURN_IN = 150; Sampler_lag = 10;
llh = [];
% repeat experiement 3 times
for i = 1:4
[theta, phi, avetheta, avephi, L] = ...
    gibbs(alpha, beta, Doc, T, W, Iter, BURN_IN, Sampler_lag);
    llh = [llh, L];
end
% Plot loglikelihood
%{
showlist = [10,20,50,150,300,400,500];
for i = 1:3
    n = length(llh(:,i));
    plot(llh,1:n,'-k'); hold on
    plot()
end
%}
% show learned topic
%{
figure
for i = 1:10
    subplot(2,5,i);
    im = reshape(phi(:,i),5,5);
    imagesc(im), colormap gray, title(['topic',num2str(i)]);
end
%}
