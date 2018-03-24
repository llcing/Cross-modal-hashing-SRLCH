function [p, r] = evaluate_micro(Rel, Ret)
% evaluate micro_averaged performance
% Input:
%    Rel = relevant  train documents for each test document
%    Ret = retrieved train documents for each test document
% Output:
%    p   = micro-averaged precision
%    r   = micro-averaged recall


relevant_num = sum(Rel(:));
 
retrieved_relevant_num = sum(Rel(Ret));
%retrieved_relevant_num = (Rel & Ret);  

retrieved_num = sum(Ret(:));

p = retrieved_relevant_num/retrieved_num;

r= retrieved_relevant_num/relevant_num;


end
