%% This will use an off the shelf solver to minimize diversity, using the phylogenetic, identity, and random similarity matricies (along with Quikr)
k_size = 3;  % k-mer size
addpath(genpath('Data'))
file = sprintf('97_otus_subset.fasta_A_%d.mat', k_size);
A_k = load(file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the precomputed data if you don't want to wait for the computations to finish
if isfile('MinimizeDiversityStandardSolverData.mat')
    load('MinimizeDiversityStandardSolverData.mat')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% reduce the size of the problem
N = min([3*4^k_size, 10000]);
A_k = full(A_k.A_k(:,1:N));


%% Import the Z matrices
D = load('97_otus_subset_phylogenetic_distance.mat');  % the phylogenetic similarity matrix
D = D.D;
D = D(1:N, 1:N);  % reduce to relevant number of organisms
k = 5; %sqp, k_size 3, 1->20 support sizes
Z_phylo = exp(-k.*D);  % phylogenetic similarity matrix

B = (A_k>0);  % the B matrix from the paper
Z_BA = B'*A_k;  % NOTE: for this small of a k-mer size (required due to the concave optimization being inneficient), this matrix is all 1's, so will not use
Z_I = eye(N);  % identity similarity matrix
Z_rand = rand(N);  % random similarity matrix
Z_rand = Z_rand - diag(diag(Z_rand)) + eye(N);  % make sure the diagonal is all 1's

%% Look at the correlation/L1 norm between columns of A and the entries of D (better for large k sizes)
A_k_l1columns=squareform(pdist(A_k', 'minkowski', 1));  % L1 norm between columns of A
figure();
scatter(A_k_l1columns(:), D(:),'.')  % to visualize the correspondence between the phylogenetic similarity and the L1 norm between the columns of A
title(sprintf('k=%d, corr=%f',k_size, corr(A_k_l1columns(:), D(:))))
xlabel('L1 norm between columns of A^k')
ylabel('Phylogenetic distance')

%% Construct similarity matrices: Add an epsilon to the similarity matrices so nothing goes wrong with division by zero
eps_z = 1e-5;
Z_BA = (Z_BA + eps_z*ones(N))./(1+eps_z);
Z_I = (Z_I + eps_z*ones(N))./(1+eps_z);
Z_rand = (Z_rand + eps_z*ones(N))./(1+eps_z);
Z_phylo = (Z_phylo + eps_z*ones(N))./(1+eps_z);

%% Correspondence is even better for the phylogenetic matrix (better for large k sizes)
A_k_l1columns=squareform(pdist(A_k', 'minkowski', 1));  % L1 norm between columns of A
figure();
scatter(A_k_l1columns(:), 1-Z_phylo(:),'.')  % to visualize the correspondence between the phylogenetic similarity and the L1 norm between the columns of A
title(sprintf('k=%d, corr=%f',k_size, corr(A_k_l1columns(:), 1-Z_phylo(:))))
xlabel('L1 norm between columns of A^k')
ylabel('1-Phylogenetic similarity matrix')


%% Do the optimization over all the simulations
minS = 1;  % smallest support size
maxS = 20;  % largest support size
stepSize = 2;  % step size for the supports
suppSizes = minS:stepSize:maxS;  % all the support sizes
numReps = 200;  % number of replicates to do
q = .01;  % diversity q to use

% matrices to store the mean l1 errors
phylo_norms = zeros(1,length(suppSizes));
phylo_plain_norms = zeros(1,length(suppSizes));
ID_norms = zeros(1,length(suppSizes));
%BA_norms = zeros(1,length(suppSizes));
rand_norms = zeros(1, length(suppSizes));
quikr_norms = zeros(1, length(suppSizes));

% matrices to store the l1 errors per replicates
errors_phylo = zeros(numReps, length(suppSizes));
errors_phylo_plain = zeros(numReps, length(suppSizes));
errors_ID = zeros(numReps, length(suppSizes));
%errors_BA = zeros(numReps, length(suppSizes));
errors_rand = zeros(numReps, length(suppSizes));
errors_quikr = zeros(numReps, length(suppSizes));

% optimization parameters
MaxFunctionEvaluations = 3000;
OptimalityTolerance = .0001;
ConstraintTolerance = .0001;
FunctionTolerance = .00001;
UseParallel = false;

tic;
for suppSizeInd=1:length(suppSizes)
    suppSize = suppSizes(suppSizeInd);
    
    % to store the erros per support size
    temp_phylo = zeros(1, numReps);
    temp_phylo_plain = zeros(1, numReps);
    temp_ID = zeros(1, numReps);
    %temp_BA = zeros(1, numReps);
    temp_rand = zeros(1, numReps);
    temp_quikr = zeros(1, numReps);
    rep_ind = 0;
    fprintf('On support size %d of %d\n', suppSizeInd, length(suppSizes));
    parfor rep=1:numReps
        supp = datasample(1:N, suppSize, 'Replace', false);
        % normalized uniformly random true x vector
        profile_x = zeros(N,1);
        profile_x(supp) = rand(suppSize,1);
        profile_x = profile_x./sum(profile_x);
        
        % measurement vector y
        y = A_k*profile_x;
        
        x0 = B'*y;  % Initial point based off of B'
        x0 = x0./sum(x0);  % Initial point based off of B'
                
        % phylogenetic without any hessian or grad
        Z = Z_phylo;
        options = optimoptions('fmincon','SpecifyObjectiveGradient',false,'Algorithm','sqp','Display','off',...
        'Diagnostics','off', 'MaxFunctionEvaluations', MaxFunctionEvaluations, 'UseParallel',UseParallel, 'OptimalityTolerance', OptimalityTolerance,'ConstraintTolerance', ConstraintTolerance,...
        'FunctionTolerance',FunctionTolerance,'SpecifyConstraintGradient',false);
        [x_phylo_plain, fval, exitflag, output] = fmincon(@(x)diversity_pow(x, q, Z), x0, [], [], A_k, y, zeros(1,N), ones(1,N), [], options);
        fprintf('phylo_plain %f\n', sum(x_phylo_plain))  % print out the norm before normalizing just ot check everything is working as it should
        
        % phylogenetic similarity matrix
        Z = Z_phylo;
        options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Algorithm','sqp','Display','off',...
        'Diagnostics','off', 'MaxFunctionEvaluations', MaxFunctionEvaluations, 'UseParallel',UseParallel, 'OptimalityTolerance', OptimalityTolerance,'ConstraintTolerance', ConstraintTolerance,...
        'FunctionTolerance',FunctionTolerance, 'FiniteDifferenceStepSize', .00001, 'StepTolerance', .0001, 'MaxIterations', 5); % 'SpecifyConstraintGradient',true, % don't know why I was specifying the constraint gradient
        [x_phylo, fval, exitflag, output, lambda, grad, hess] = fmincon(@(x)diversity_pow_grad(x, q, Z), x0, [], [], A_k, y, zeros(1,N), ones(1,N), [], options);
        fprintf('phylo %f\n', sum(x_phylo))  % print out the norm before normalizing just ot check everything is working as it should
        
        % identity similarity matrix
        Z = Z_I;
        [x_ID, fval, exitflag, output, lambda, grad, hess] = fmincon(@(x)diversity_pow_grad(x, q, Z), x0, [], [], A_k, y, zeros(1,N), ones(1,N), [], options);
        fprintf('ID %f\n', sum(x_ID))  % print out the norm before normalizing just ot check everything is working as it should
        
        % unused B*A similarity matrix
        %Z = Z_BA;
        %[x_BA, fval, exitflag, output, lambda, grad, hess] = fmincon(@(x)diversity_pow_grad(x, q, Z), x0, [], [], A_k, y, zeros(1,N), ones(1,N), [], options);
        %fprintf('BA %f\n', sum(x_BA))
        
        % random similarity matrix
        Z = Z_rand;
        [x_rand, fval, exitflag, output, lambda, grad, hess] = fmincon(@(x)diversity_pow_grad(x, q, Z), x0, [], [], A_k, y, zeros(1,N), ones(1,N), [], options);
        fprintf('rand %f\n', sum(x_rand))  % print out the norm before normalizing just ot check everything is working as it should
        
        % quikr
        lambda = 10000;
        x_quikr = lsqnonneg([ones(1, size(A_k,2)); lambda*A_k], [0;lambda*y]);
        fprintf('quikr %f\n', sum(x_quikr))
        
        temp_phylo(rep) = sum(abs(x_phylo./sum(x_phylo) - profile_x));
        temp_phylo_plain(rep) = sum(abs(x_phylo_plain./sum(x_phylo_plain) - profile_x));
        temp_ID(rep) = sum(abs(x_ID./sum(x_ID) - profile_x));
        temp_rand(rep) = sum(abs(x_rand./sum(x_rand) - profile_x));
        temp_quikr(rep) = sum(abs(x_quikr./sum(x_quikr) - profile_x));
    end
    
    phylo_norms(suppSizeInd) = mean(temp_phylo);
    phylo_plain_norms(suppSizeInd) = mean(temp_phylo_plain);
    ID_norms(suppSizeInd) = mean(temp_ID);
    %BA_norms(suppSizeInd) = mean(temp_BA);
    rand_norms(suppSizeInd) = mean(temp_rand);  
    quikr_norms(suppSizeInd) = mean(temp_quikr);
    
    errors_phylo(:,suppSizeInd) = temp_phylo;
    errors_phylo_plain(:,suppSizeInd) = temp_phylo_plain;
    errors_ID(:,suppSizeInd) = temp_ID;
    %errors_BA(:,suppSizeInd) = temp_BA;
    errors_rand(:,suppSizeInd) = temp_rand;
    errors_quikr(:,suppSizeInd) = temp_quikr; 
end
fprintf('Finished\n')
toc
%% L1 norm error plots
figure();
hold on
plot(suppSizes, phylo_norms, 'r')
plot(suppSizes, ID_norms, 'g')
%plot(suppSizes, BA_norms, 'b')
plot(suppSizes, rand_norms, 'k')
plot(suppSizes, quikr_norms, 'y')
legend('phylo', 'Ident', 'BA', 'random','Quikr')
xlabel('support size')
ylabel('L1 error')
set(gca,'DataAspectRatio',[15 1.5 1])
%x= 1.5;
%set(gcf,'Position',[x*100 x*100 x*500 x*500])
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];

saveas(gcf, 'Figures/L1ErrorFmincon.png')
%% Vizualization: Percent recovered plot:
% The percentage (over all replicates) of successful recoveries as a 
% function of the support size
thresh = 1e-4;  % anything L1 error smaller than this is considered "successful"

colors = linspecer(6);
fs = 15;
set(groot,'defaultAxesColorOrder', [0 0 0], 'DefaultAxesLineStyleOrder','-|--|:|-.|:*|:^')
line_width = 2;
fig = figure();
hold on
plot(minS:stepSize:maxS, mean(errors_phylo<thresh), 'LineWidth', line_width, 'Color', colors(1,:))
%plot(minS:stepSize:maxS, mean(errors_phylo_plain<thresh), 'LineWidth', line_width, 'Color', colors(2,:))
plot(minS:stepSize:maxS, mean(errors_ID<thresh), 'LineWidth', line_width, 'Color', colors(3,:))
%plot(minS:stepSize:maxS, mean(errors_BA<thresh), 'LineWidth', line_width, 'Color', colors(4,:))
plot(minS:stepSize:maxS, mean(errors_rand<thresh), 'LineWidth', line_width, 'Color', colors(5,:))
plot(minS:stepSize:maxS, mean(errors_quikr<thresh), 'LineWidth', line_width, 'Color', colors(6,:))
ylabel('Percent recovered', 'FontSize', fs)
xlabel('Support size', 'FontSize', fs)
%title(sprintf('Reconstruction performance: percent recovered norm using k = %d', k))
%lgd = legend('phylo','phylo plain', 'Ident', 'BA', 'random','Quikr');
lgd = legend('Phylogenetic','Identity', 'Random','Quikr');
lgd.FontSize = fs;
set(gca,'DataAspectRatio',[15 1.5 1])
%x= 1.5;
%set(gcf,'Position',[x*100 x*100 x*500 x*500])
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf, 'Figures/PercentRecoveredFmincon.png')

%% export the data
if ~isfile('MinimizeDiversityStandardSolverData.mat')
    save('MinimizeDiversityStandardSolverData.mat')
end