% This script generate 10 predefined images
close all
clear
% @image 5-by-5 pixels, range from 0~1
[x,y] = meshgrid(1:5,1:5);
X = x(:); Y = y(:);

topic = cell(10,1);

% Pattern 1 
pat = zeros(5,5);
pat(Y(X == 1),X(X == 1)) = 1;
topic{1,1} = pat/sum(pat(:));

% Pattern 2
pat = zeros(5,5);
pat(Y(X == 2),X(X == 2)) = 1;
topic{2,1} = pat/sum(pat(:));

% Pattern 3 
pat = zeros(5,5);
pat(Y(X == 3),X(X == 3)) = 1;
topic{3,1} = pat/sum(pat(:));

% Pattern 4 
pat = zeros(5,5);
pat(Y(X == 4),X(X == 4)) = 1;
topic{4,1} = pat/sum(pat(:));

% Pattern 5
pat = zeros(5,5);
pat(Y(X == 5),X(X == 5)) = 1;
topic{5,1} = pat/sum(pat(:));

% Pattern 6 
pat = zeros(5,5);
pat(Y(Y == 1),X(Y == 1)) = 1;
topic{6,1} = pat/sum(pat(:));

% Pattern 7 
pat = zeros(5,5);
pat(Y(Y == 2),X(Y == 2)) = 1;
topic{7,1} = pat/sum(pat(:));

% Pattern 8 
pat = zeros(5,5);
pat(Y(Y == 3),X(Y == 3)) = 1;
topic{8,1} = pat/sum(pat(:));

% Pattern 9 
pat = zeros(5,5);
pat(Y(Y == 4),X(Y == 4)) = 1;
topic{9,1} = pat/sum(pat(:));

% Pattern 10 
pat = zeros(5,5);
pat(Y(Y == 5),X(Y == 5)) = 1;
topic{10,1} = pat/sum(pat(:));

% clear up workspace and save
clear x y X Y
clear pat
save('predefinedtopic.mat');

