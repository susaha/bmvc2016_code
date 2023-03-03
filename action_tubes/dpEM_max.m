% -------------------------------------------------------------------
% Author: Gurkirt Singh
% second pass dynamic programming
% -------------------------------------------------------------------
function [p,q,D] = dpEM_max(M,alpha)
[r,c] = size(M);
D = zeros(r, c+1); % add an extra column
D(:,1) = 0; % put the maximum cost
D(:, 2:(c+1)) = M;
v = [1:r]';
phi = zeros(r,c);
for j = 2:c+1; 
    for i = 1:r; 
        [dmax, tb] = max([D(:, j-1)-alpha*(v~=i)]);
        D(i,j) = D(i,j)+dmax;
        phi(i,j-1) = tb;
    end
end
D = D(:,2:(c+1));
q = c; 
[~,p] = max(D(:,c));
i = p; 
j = q; 
while j>1 
    tb = phi(i,j);
    p = [tb,p];
    q = [j-1,q];
    j = j-1;
    i = tb;
end