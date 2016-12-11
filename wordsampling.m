% This function smaples 100 words for one document
% @param topic
%	predefined topic, T-by-1 cell array 
% @param theta
%	topic distribution for this document, 1-by-T vector
% @return doc
%	sampling document, 1-by-100 vector
function doc = wordsampling(topic,theta)
    T = length(topic);
    W = length(topic{1,1}(:)); % get size of vocabulary
    % calculate words distribution for this document
    pword = zeros(1,W);
    for w = 1:W
	for t = 1:T
	    pword(w) = pword(w) + topic{t,1}(w)*theta(t); 
	end
    end

    % cumulate word distribution
    for w = 2:w
	pword(w) = pword(w) + pword(w-1);    
    end

    % sample 100 words from cumulative multinomial distribution
    % pword
    doc = zeros(1,100);
    for i = 1:100
	u = rand*pword(w);
	for word = 1:W
	    if(u < pword(word))
		break;
	    end
	end
	doc(1,i) = word;
    end

end
