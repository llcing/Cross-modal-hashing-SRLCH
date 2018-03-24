% EVALUATION
% Flag: 
% 0 -- compute all 
% 1 -- MAP only
% 2 -- prerank & recrank only
% 3 -- pno & rno only

%function results = evaluation(H,tH,cateTrainTest)
function [MAP, prerank, recrank, pno, rno] = evaluation(H,tH,cateTrainTest)
% Set flag
flag = 1;
MAP = 0;
prerank = 0;
recrank = 0;
pno = 0;
rno = 0;

addpath('./eval');

B = compactbit(H); 
tB = compactbit(tH);
Dhamm = hammingDist(tB,B);

hammTrainTest = Dhamm' ;
[~, HammingRank] = sort(hammTrainTest,1) ;

% set M_set to compute precision recall @K
% e.g. M_set = [K1, K2]
M_set = [100 200 300 400 500] ;  

% Evaluation
if flag == 0 || flag == 1
    % MAP@all  
    MAP = cat_apcal_similarity(cateTrainTest,HammingRank) ; 
end

if flag == 0 || flag == 2
    % precision and reall at each data point [macro]
    [prerank, recrank] = evaluate_HammingRanking_category_similarity(cateTrainTest, HammingRank) ; 
end

if flag == 0 || flag == 3
    % precision and recall @K [macro]
    [pno, rno] = cat_ap_topK(cateTrainTest,HammingRank, M_set) ;  
end
 
