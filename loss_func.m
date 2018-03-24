% compute the loss value
function loss = loss_func(B, V, T, Y, PV, PT, W, muv, mut)
loss = norm(B-Y*W,'fro').^2+muv.*norm(B-V*PV,'fro').^2+mut.*norm(B-T*PT,'fro').^2;
%loss = norm(B-Y*W, 'fro').^2;