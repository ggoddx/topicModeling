d = zeros(length(Doc),25);
for i = 1:length(Doc)
    doc = Doc{i,1};
    bin = 1:25;
    h = hist(doc,bin);
    d(i,:) = h;
end
