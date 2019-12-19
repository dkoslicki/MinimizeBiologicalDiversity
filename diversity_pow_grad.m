function [f,g] = diversity_pow_grad(x,q,Z)
% [f,g] = diversity_pow_grad(x,q,Z) computes the diversity (f) of the vector x using
% order q and a similarity matrix Z, as well as the gradient (g). NOTE: either all entries of Z must be
% strictly positive, or else you must be careful to only apply to x's and
% Z's such that (Z*x)_i>0 for all i.
% one approach is to take:
% Z -> (Z+\epsilon*all_ones)/(1+\epsilon)
    f = sum(x.*(Z*x).^(q-1)); % this is the diversity ||x||_{Z,q}^q
    g = (Z*x).^(q-1)-(1-q)*Z'*(x.*(Z*x).^(q-2)); % this has been checked and is correct
end
