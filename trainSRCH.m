% CODES OF SRCH_TRAINING
% AUTHOR: LUCHEN LIU
% para -- the parameter variable
% V -- visual feature
% T -- textual feature
% Y -- label information
% nbits -- the number of hash bits
% Ntrain -- the number of samples
% show_loss -- if True print loss

function [B, PV, PT, loss] = trainSRCH(V, T, Y, para, nbits, Ntrain, show_loss)

lam = para.lam; % lambda for W term
alp = para.alp; % alpha for PV term
bet = para.bet; % beta for PT term
maxItr = para.maxItr; % max iteration
muv = para.muv;
mut = para.mut;

loss=[];
disp('Begin Training...');
% Initialization
randn('seed', 10);
PV = randn(Ntrain, nbits);
PT = randn(Ntrain, nbits);
W = randn(Ntrain, nbits); % W is the projection matrix of Y
B = sign(randn(Ntrain,nbits));

% Temporary variables to speed up the algorithm
PVtmp = (V'*V+alp.*eye(size(V,2)))\V';
PTtmp = (T'*T+bet.*eye(size(T,2)))\T';
Wtmp = (Y'*Y+lam.*eye(size(Y,2)))\Y';

i = 0;
while i < maxItr

    % Step-1
    PV = PVtmp*B;
    PT = PTtmp*B;
    % Step-2
    W = Wtmp*B;
    % Step-3
    tmpV = V*PV;
    tmpT = T*PT;
    tmpYW = Y*W;
    B = sign(tmpYW + muv.*tmpV + mut.*tmpT);
    
    i = i+1;

    % show_loss
    if show_loss
        loss(i)=loss_func(B, V, T, Y, PV, PT, W, muv, mut);
        fprintf('Iteration  %03d: %d\n',i, loss(i));
    else
        fprintf('Iteration  %03d\n',i);
    end
end
disp('Training Complete!');