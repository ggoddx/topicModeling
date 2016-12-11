% This function smaples a multinomial distribution from Dirichlet distribution
% @param alpha
%	Dirichlet parameter, 1-by-p vector
% @param n
%	number of sampling
% @return r
%	n-by-p sampling matrix
function r = drchrnd(alpha,n)
    p = length(alpha);
    r = gamrnd(repmat(alpha,n,1),1,n,p);
    r = r./ repmat(sum(r,2),1,p);
end

