function d = diversity_pow(x,q,Z)
% d = diversity_pow(x,q,Z) computes the diversity of the vector x using
% order q and a similarity matrix Z. NOTE: either all entries of Z must be
% strictly positive, or else you must be careful to only apply to x's and
% Z's such that (Z*x)_i>0 for all i.

d = sum(x.*(Z*x).^(q-1));
end
