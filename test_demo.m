% CODES OF SRLCH_TRAINING
% AUTHOR: LUCHEN LIU
% test demo
% just a demo
clear all;


% Load dataset
% label_encode_flag is to control whether encoding the class labels to one-hot format
norm_flag = 1; % control whether to normalization

%dataset = 'uci_all'; label_encode_flag = 1;
%dataset = 'mirflickr25k'; label_encode_flag = 0;
dataset = 'wikiData'; label_encode_flag = 1;
%dataset = 'nus_wide2k'; label_encode_flag = 0;
%dataset = 'labelme_sigir14'; label_encode_flag = 1;

if exist(['./testbed/', dataset, '.mat'],'file')
    load(['./testbed/', dataset]);
else
    prepare_dataset(dataset, label_encode_flag, norm_flag);
    load(['./testbed/', dataset]);
end

%bits=[16 32 64 128]; % bits of hash code
bits=[32];

% Set parameter 
% Iteration numbers:
%   wiki: 5
%   uci: 8
%   mirflickr: 50
%   nus_wide: 30-40
%   labelme: 5-7
para.maxItr = 10;
para.muv = 1e-1;
para.mut = 1e-5;
para.lam = 0.1;
para.alp = 0.01;
para.bet = 0.1;

Ntrain = size(I_tr, 1); % Train samples
Ntest = size(I_te, 1); % Test samples

% Kernelization
switch dataset
    case 'nus_wide2k'
        n_anchors = 5000;
    otherwise
    n_anchors = 500;
end

n_anchor = 100;
[I_tr, I_te, T_tr, T_te] = Phi_kernel_all(I_tr, I_te, T_tr, T_te, Ntrain, Ntest, n_anchors);

% Begin training
show_loss = 1;
for bi=1:length(bits)
    nbits=bits(bi);
    % Train
    tic
    [H, PV, PT,loss{bi}] = trainSRCH(I_tr, T_tr, L_tr, para, nbits, Ntrain, show_loss);
    toc;


    % Img2Txt
    tHI = I_te*PV;

    % Txt2Img
    tHT = T_te*PT;
    
    % Hash codes generation
    H = H > 0;
    tHI = tHI > 0;
    tHT = tHT > 0;

    % Evaluation
    [results{bi}.MAP1,results{bi}.PreRank1,results{bi}.RecRank1,results{bi}.PreNo1,results{bi}.RecNo1]=evaluation(H,tHI,cateTrainTest);
    [results{bi}.MAP2,results{bi}.PreRank2,results{bi}.RecRank2,results{bi}.PreNo2,results{bi}.RecNo2]=evaluation(H,tHT,cateTrainTest);


    % Show the results of mAP
    % Img2Txt
    results{bi}.MAP1
    % Txt2Img
    results{bi}.MAP2

end

% Save all the results
save_path = './results'; 
data_name = fullfile(sprintf('%s/results_%s.mat', save_path, dataset));
save(data_name, 'results', 'loss');

