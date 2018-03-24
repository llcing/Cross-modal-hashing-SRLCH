function prepare_dataset(dataset, label_encode_flag, norm_flag)
% dataset is stored in a row-wise matrix
%%
load(['./dataset/',dataset]);

I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

% Label encoding
if label_encode_flag
    L_tr_ = label_encode(L_tr);
    L_te_ = label_encode(L_te);
    L_tr = L_tr_;
    L_te = L_te_;
    clear L_tr_, L_te_;
end 
% Normalize all feature vectors to unit length
if norm_flag
    I_tr = normalize(I_tr);
    I_te = normalize(I_te);
    T_tr = normalize(T_tr);
    T_te = normalize(T_te);
end

cateTrainTest = L_tr*L_te' > 0;

save(['./testbed/',dataset],'I_tr','I_te','T_tr','T_te','L_tr','L_te','cateTrainTest');

clear;