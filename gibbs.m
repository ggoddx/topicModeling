function [theta, phi, avetheta, avephi, L] = gibbs(alpha, beta, Doc, T, W, Iter, BURN_IN, Sampler_lag)
% This function implements gibbs sampling
% This gibbs sampler is defined as a class
% @param W
%	size of vocabulary
% @param T	
%       number of topics
% @param Doc	
%	Docment list
% @param alpha, beta
%	Dirichlet Distribution parameter
% @param
%	Iter:= iterations
%	BURN_IN:= number of iteratoins required before Markov Chain reaches equilibrium 
%	Sampler_lag:= interval between two samplings

% @return theta, avetheta
%	topic distribution for documents, D-by-T matrix
% @return phi, avephi
%	words distribution in each topics, W-by-T matrix

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
    phi = zeros(W,T);
    thetasum = zeros(D,T);
    phisum = zeros(W,T);
    % loglikelihood
    L = [];
    lconst = T*(gammaln(W*beta) - W*gammaln(beta)); % constant for calculating L
    tmp = LogLikelihood;
    L = [L;tmp];
    % number of times updating parameter
    numstat = 0;
    % iteration parameter
    %BURN_IN = 100;
    showlist = [10,20,50,150,300,400,500];
    displaycount = 1;
    l = length(showlist);
    figure;
    for i = 1:Iter
        if (mod(i,100) == 0)
            disp([num2str(i),'th Iteration'])
        end
        for d = 1:D
                N = length(Doc{d,1});
            for w = 1:N
                z{d,1}(w) = PosteriorSampling(d,w,z);
            end
        end   
        %{
        if ((i >= BURN_IN && Sampler_lag > 0) && mod(i,Sampler_lag) == 0)
            [theta, phi] = UpdateParam;
            thetasum = thetasum + theta;
            phisum = phisum + phi;
        end
        %}
        % For the purpose of experiment
        % record parameter
        % visualize document
        % calculate loglikelihood and visualize
        if (mod(i,Sampler_lag) == 0)
            % update parameter
            [theta, phi] = UpdateParam;
            thetasum = thetasum + theta;
            phisum = phisum + phi;  
            % update loglikelihood
            tmp = LogLikelihood;
            L = [L;tmp];
        end
        if ( i == showlist(displaycount) )
            for j = 1:10
                subplot(l,10,j+10*(displaycount-1));
                im = reshape(phi(:,j),5,5);
                imagesc(im), colormap gray%, title(['topic',num2str(j)]);
            end
            %suptitle([num2str(i),'iteration']);
            displaycount = displaycount + 1;
        end
    end
    avetheta = thetasum/numstat;
    avephi = phisum/numstat;

    % @brief Initialize Markoc Chain
    % @param T
    %	  number of topic
    function assignment = InitialMarkovChain(T)
        assignment = cell(D,1);
        for docidx = 1:D
            n = length(Doc{docidx,1});
            % ndsum:= number of words assigned in document d
            ndsum(docidx) = n;
            assignment{docidx,1} = zeros(1,n);
            for wordidx = 1:n   
                % randomly initialize topic assigment
                t = ceil(rand*T);
                assignment{docidx,1}(wordidx) = t;
                %keyboard;
                % nw:= number of times word i have been assigned to Topic j
                nw(Doc{docidx,1}(wordidx),t) = nw(Doc{docidx,1}(wordidx),t) + 1;
                % nd:= number of times Topic j has been assigned to Document i
                nd(docidx,t) = nd(docidx,t) + 1; 
                % nwsum:= total number of words assgined to Topic j
                nwsum(t) = nwsum(t) + 1;
            end
        end	
    end

    % @brief sample topic assginment for Word w in Document d
    % @param d
    %	  document
    % @param w
    % 	  word
    function assignment = PosteriorSampling(docidx,wordidx,currentAssignment)
        % First remove current assigment of Word w from the count
        currentTopic = currentAssignment{docidx,1}(wordidx);
        vocidx = Doc{docidx,1}(wordidx);
        nw(vocidx,currentTopic) = ...
                            nw(vocidx,currentTopic) - 1;
        nd(docidx,currentTopic) = nd(docidx,currentTopic) - 1;
        nwsum(currentTopic) = nwsum(currentTopic) - 1;
        ndsum(docidx) = ndsum(docidx) - 1;
        % Second calculate multinomial posterior conditioned on 
        % all the other assignments and cumulate this distribution
        prob = zeros(T,1);
        for t = 1:T
            prob(t) = (nw(vocidx,t) + beta)/(nwsum(t) + W*beta) *...
                      (nd(docidx,t) + alpha)/(ndsum(docidx) + T*alpha); 
            if (t >= 2)
                prob(t) = prob(t) + prob(t-1);
            end
        end
        % Third sample from this distribution 
        u = rand*prob(T);
        for assignment = 1:T
            if(u < prob(assignment))
                break;
            end
        end
        % Finally update count
        nw(vocidx,assignment) = nw(vocidx,assignment) + 1;
        nd(docidx,assignment) = nd(docidx,assignment) + 1;
        nwsum(assignment) = nwsum(assignment) + 1;
        ndsum(docidx) = ndsum(docidx) + 1;
    end

    % @brief update topic distribution theta and topic phi, 
    %        and record number of updates
    % @theta: D-by-T matrix
    % @phi: W-by-T matrix
    function [theta, phi] = UpdateParam
        theta = zeros(D,T);
        phi = zeros(W,T);
        for t = 1:T
            for docidx = 1:D
                theta(docidx,t) =...
                    (nd(docidx,t) + alpha)/(ndsum(docidx) + T*alpha);
            end
            for wordidx = 1:W
                phi(wordidx,t) =...
                    (nw(wordidx,t) + beta)/(nwsum(t) + W*beta);
            end
        end
            numstat = numstat + 1;
    end
    
    % @brief calculate loglikelihood log(p(w|z))
    function l = LogLikelihood
        ltmp = sum(sum(gammaln(nw + beta),1) - gammaln(nwsum' + W*beta)); 
        l = lconst + ltmp;
    end
end









