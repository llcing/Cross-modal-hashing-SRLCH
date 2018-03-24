% Label encoding 
% Encoding label to one-hot vector
% for sigle label 
% TODO: Multilabel
function L_en = label_encode(L)
N = size(L, 1);
cate = max(L(:,1));
L_en = zeros(N, cate);
for i = 1 : N 
    L_en(i, L(i))=1;
end

% or a effective way:
% Y = sparse(1:length(y), double(y), 1); Y = full(Y);