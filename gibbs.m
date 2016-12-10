function [theta, phi, avetheta, avephi] = gibbs(alpha, beta, Doc, T, W, Iter, BURN_IN, Sampler_lag)
% This function implements gibbs sampling
% This gibbs sampler is defined as a class
% member variable : 
%	W:= size of vocabulary
%	T:= number of topics
%	Doc:= Docment list
%	D:= Total number of documents
%	alpha, beta:= Dirichlet Distribution parameter
% optional:
%	Iter:= iterations

% Step 1. 
% 	Define z to be a D-by-W matrix 
% 	Initialize each rows of z independently in the range of 1:K  using rand()
 
% Step 2.
%	Iterate through z matrix 
%	For each entry of z, update conditioned on other entries
%	Detail: 1) remove zij from current count
%		2) calculate conditional multinomial distribution
%		3) compute cumulative distribution
%		4) sample from uniform distribution

% Step 3.
%	Op1. store every theta and phi (in the form of summation) aftre Burn-in period
% 	When retrieved, return mean value of theta and phi
%	Op2. When retrived, calculate theta and phi directly from current sampling

    D = length(Doc);
    % ndsum:= number of words assigned in document d
    % nd:= number of times Topic j has been assigned to Document i
    % nwsum:= total number of words assgined to Topic j
    % nw:= number of words i assigned to Topic j
    ndsum = zeros(D,1);
    nwsum = zeros(T,1);
    nw = zeros(W,T);
    nd = zeros(D,T);
    % z topic assginment list
    z = InitialMarkovChain(T);
    % thetasum:= sum of topic distribution for each document 
    % phisum:= sum of words distribution for each topic
    theta = zeros(D,T);
    phi = zeros(T,W);
    thetasum = zeros(D,T);
    phisum = zeros(T,W);
    % number of times updating parameter
    numstat = 0;
    % iteration parameter
    BURN_IN = 100;
    for i = 1:Iter
	for d = 1:D
            N = length(Doc{d,1});
	    for w = 1:N
		z{d,1}(w) = PosteriorSampling(d,w,z);
	    end
	end    
	if ((i >= BURN_IN && Sampler_lag > 0) && mod(i,Sampler_lag) == 0)
	    theta = Updatetheta();
	    phi = Updataphi();
	    thetasum = thetasum + theta;
 	    phisum = phisum + phi;
	end
    end
    avetheta = thetasum/numstat;
    avephi = phisum/numstat;
end

% @brief Initialize Markoc Chain
% @param T
%	  number of topic
function z = InitialMarkovChain(T)
    z = cell(D,1);
    for d = 1:D
	N = length(Doc{d,1});
	% ndsum:= number of words assigned in document d
        ndsum(d) = N;
	for word = 1:N   
	    % randomly initialize topic assigment
	    topic = floor(rand(1,N)*T + 0.5);
	    z{d,1}(word) = topic;
	    % nw:= number of times Words i have been assigned to Topic j
	    nw(Doc{d,1}(word),topic) = nw(Doc{d,1}(word),topic) + 1;
	    % nd:= number of times Topic j has been assigned to Document i
    	    nd(d,topic) = nd(d,topic) + 1; 
	    % nwsum:= total number of words assgined to Topic j
            nwsum(topic) = nwsum(topic) + 1;
	end
    end	
end

% @brief sample topic assginment for Word w in Document d
% @param d
%	  document
% @param w
% 	  word
function assignment = PosteriorSampling(d,w)
    % First remove current assigment of Word w from the count
    currentTopic = z{d,1}(w);
    nw(Doc{d,1}(w),currentTopic) = nw(Doc{d,1}(w),currentTopic) - 1;
    nd(d,currentTopic) = nd(d,currentTopic) - 1;
    nwsum(currentTopic) = nwsum(currentTopic) - 1;
    ndsum(d) = ndsum(d) - 1;
    % Second calculate multinomial posterior conditioned on all the other assignments
    % and cumulate this distribution
    for topic = 1:T
    	prob(topic) = (nw(Doc{d,1}(w),currentTopic) + beta)/(nwsum(currentTopic) + W*beta) *...
		      (nd(d,currentTopic) + alpha)/(ndsum(d) + T*alpha) ; 
	if (topic >= 2)
	    prob(topic) = prob(topic) + prob(topic-1);
	end
    end
    % Third sample from this distribution 
    u = rand*prob(T);
    for assignment = 1:T
	if(u < prob(assignment)
	    break;
	end
    end
    % Finally update count
    nw(Doc{d,1}(w),assignment) = nw(Doc{d,1}(w),assignment) + 1;
    nd(d,assignment) = nd(d,assignment) + 1;
    nwsum(assignment) = nwsum(assignment) + 1;
    ndsum(d) = ndsum(d) + 1;
end

% @brief update topic distribution theta and topic phi, and record number of updates
% @theta: D-by-T matrix
% @phi: W-by-T matrix
function phi = UpdateParam()
    theta = zeros(D,T);
    phi = zeros(W,T);
    for topic = 1:T
	theta(:,topic) = (nd(:,topic) + alpha)./(ndsum + T*alpha);
	phi(:,topic) = (nw(:,topic) + beta)/(nwsum(topic) + W*beta);
    end
    numstat = numstat + 1;
end










