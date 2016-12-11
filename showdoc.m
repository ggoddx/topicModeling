% This function assume the size of vocabulary is 25, and
% viusalize one document in a 5-by-5 image. 
% Pixel intensity stands for the frequency of the corresponding
% word in the document
% @param doc
%	document, 1-by-N vector
function showdoc(doc)
    N = length(doc);
    % count word frequency
    bin = 1:25;
    h = hist(doc,bin);
    h = h/sum(h);
    % show document image
    im = reshape(h,5,5);
    imagesc(im), colormap gray;
    axis square
end
