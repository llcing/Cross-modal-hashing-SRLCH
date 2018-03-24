% Evaluation_demo

addpath('./eval')
% get Hamming distance between queries and database
B = compactbit(H); 
tB = compactbit(tH);
Dhamm = hammingDist(tB,B);

hammRadius = 2 ;
hammTrainTest = Dhamm' ;
[~, HammingRank] = sort(hammTrainTest,1) ;
Ret = (hammTrainTest <= hammRadius+0.00001) ;

% set M_set to compute precision recall @K
% e.g. M_set = [K1, K2]
M_set = [500] ;                      

% ---------------------------------- %
% 用标签信息作为 groundtruth
% semantic labels as groundth

% if without_cateTrainTest and isnot_multilabel, compute MAP with traingnd and testgnd.
without_cateTrainTest = 0; 


% hash lookup: precision and reall

% *** NOT GENERAL *** %
% precision and reall at each hamming distance [micro]
[recall, precision, rate] = recall_precision(cateTrainTest', hammTrainTest') ;          
    
% hamming distance -> precision and recall [macro]
[Pre, Rec] = evaluate_macro(cateTrainTest, Ret) ;  
    
% hamming distance -> precision and recall [micro]
[Prei, Reci] = evaluate_micro(cateTrainTest, Ret) ;                                    

% *** GENERAL *** %
if without_cateTrainTest
    % precision and reall at each data point [macro]
    [prerank, recrank] = evaluate_HammingRanking_category(traingnd, testgnd, HammingRank) ; 
else
    % precision and reall at each data point [macro]
    [prerank, recrank] = evaluate_HammingRanking_category_similarity(cateTrainTest, HammingRank) ; 
end 

% *** GENERAL *** %
% hamming ranking: MAP    
if without_cateTrainTest
    % MAP@all 
    MAP = cat_apcal(traingnd,testgnd,HammingRank) ;                                         
else
    % MAP@all  
    MAP = cat_apcal_similarity(cateTrainTest,HammingRank) ;                                         
end 

% MAP@5000 if isnot_multilabel
MAP5k = cat_apcal_5k(traingnd,testgnd,HammingRank) ;                                    

% *** GENERAL *** %
% precision and recall @K [macro]
[pno,rno] = cat_ap_topK(cateTrainTest,HammingRank, M_set) ;                            


% ----------------------------------- %
% 用欧式距离作为 groundtruth
% Euclidean neighbors as groundth (NOT GENERAL)

% hash lookup: precision and reall
[recallE, precisionE, rateE] = recall_precision(WtrueTestTraining, hammTrainTest');
    
[PreE, RecE] = evaluate_macro(WtrueTestTraining', Ret) ;
[PreiE, ReciE] = evaluate_micro(WtrueTestTraining', Ret) ;

% hamming ranking: MAP
[prerankE, recrankE] = evaluate_HammingRanking_category_similarity(WtrueTestTraining',HammingRank) ;
        
MAPE = cat_apcal_similarity(WtrueTestTraining', HammingRank) ;
    
% MAPE5k = cat_apcal_similarity_5k(WtrueTestTraining', HammingRank) ;  
[pnoE, rnoE] = cat_ap_topK(WtrueTestTraining',HammingRank, M_set) ;

