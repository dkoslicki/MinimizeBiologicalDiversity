function hessian = hessianfcn(x, q, Z, lambda)
% this is the hessian of the diversity^q applied to the vector x, using
% order q, and the similarity matrix Z. Note that lambda is a dummy
% variable and is only required for the way that MATLAB requires it for use
% in fmincon
    D = diag((Z*x).^(q-2));
    D_prime = diag(x.*(Z*x).^(q-3));
    hessian = -(1-q)*(D*Z + Z'*D - (2-q)*Z'*D_prime*Z);
end