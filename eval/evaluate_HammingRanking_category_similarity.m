function [pre, rec] = evaluate_HammingRanking_category_similarity(cateTrainTest,rank)

[numtrain, numtest] = size(rank);

precision = zeros(numtest, numtrain);
recall = zeros(numtest, numtrain);
for n = 1:numtest
    [precision(n,:), recall(n,:)] = prcal(rank(:,n),cateTrainTest(:,n));
end
pre=mean(precision);
rec=mean(recall);
end


function [Pre, Rec] = prcal(sorted_ind,label)
%%% number of true samples
num_TrueNN=sum(label);
if num_TrueNN == 0
    pause
end
%%% number of samples
numTrain=length(label);

%%% 
sorted_truefalse=(label(sorted_ind)==1);
cumsum_rank=cumsum(sorted_truefalse);

Rec =cumsum_rank/num_TrueNN;
Pre =cumsum_rank./[1:numTrain]';


end
