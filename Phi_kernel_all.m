function [I_tr, I_te, T_tr, T_te] = Phi_kernel_all(I_tr, I_te, T_tr, T_te, Ntrain, Ntest, n_anchors)

% get anchors
rand('seed',1);
anchor_I = I_tr(randsample(Ntrain, n_anchors),:);
anchor_T = T_tr(randsample(Ntrain, n_anchors),:);

% % determin rbf width sigma
% Dis = EuDist2(X,anchor,0);
% % sigma = mean(mean(Dis)).^0.5;
% sigma = mean(min(Dis,[],2).^0.5);
% clear Dis
sigma = 0.6; % for normalized data


Phi_I_tr = exp(-sqdist(I_tr,anchor_I)/(2*sigma*sigma)); 
I_tr = [Phi_I_tr, ones(Ntrain,1)];
Phi_I_te = exp(-sqdist(I_te,anchor_I)/(2*sigma*sigma)); 
I_te = [Phi_I_te, ones(Ntest,1)];

Phi_T_tr = exp(-sqdist(T_tr,anchor_T)/(2*sigma*sigma)); 
T_tr = [Phi_T_tr, ones(Ntrain,1)];
Phi_T_te = exp(-sqdist(T_te,anchor_T)/(2*sigma*sigma)); 
T_te = [Phi_T_te, ones(Ntest,1)];