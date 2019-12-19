%% Section 5: algorithm for co-occurrence similarity matrix
% In this section, we analyze the optimization procedure given in equation
% (??)

% Note: this works in MATLAB R2019a, but does not appear to work in version
% R2019b due to linprob_gurobi not playing well with R2019b

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import pre-computed data
% The following series of computations took 22 minutes on an 18 core
% i9-9980XE processor. As such, you may wish to import the following data
% and skip directly to the visualizations
if isfile('CooccurenceReproduciblesMultiData.mat')
    load('CooccurenceReproduciblesMultiData.mat')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Performe computations from scratch
% set variables
small_k = 4;  % smaller k-mer size
h_sizes = [4, 6, 13]; % h range

cols_vs_rows = 3;  % fix 3-times more columns than rows
num_species = cols_vs_rows*4^small_k;  % reduce the number of columns of 
% the sensing matrix so pictures will be generated in a reasonable amount
% of time.

%% import the data
addpath(genpath('Data'))
file = sprintf('97_otus_subset.fasta_A_%d.mat', small_k);
A_k_small = load(file);
A_k_small = A_k_small.A_k(:,1:num_species);  % sensing matrix

A_hs = {};
% compute the B matrices used to form Z = B*A
B_transpose_hs = {};
for i=1:length(h_sizes)
    h = h_sizes(i);
    file = sprintf('97_otus_subset.fasta_A_%d.mat', h);
    A_k_large = load(file);
    A_k_large = A_k_large.A_k(:,1:num_species);  % Used to form Z = B*A^{(h)}
    B_k_large = (A_k_large>0);
    B_k_large_tr = B_k_large';
    A_hs{i} = A_k_large;
    B_transpose_hs{i} = B_k_large_tr;
end


%% Simulations
% random vectors will be generated with a specified support size, and each
% of three methods will be used in an attempt to recover these vectors
% using:
% 1. Standard, unweighted L1 minimization
% 2. The optimization proceedure (??) using h = k
% 3. The optimization proceedure (??) using h > h
% Performance will be assessed in terms of L1 norm between the true vector 
% and the reconstructed vector.

% WARNING: this calculation takes a significant amount of time when
% small_k >= 4 (on the order of hours)

q = .01;  % fixed, small q value s.t. 0<q<1
start = 25;  % the starting support size
step_size = 2;  % how much to increase the support size in each step
max_support = 70;  % maximum support size
support_sizes = start:step_size:max_support;  % vector of support sizes
num_reps = 200;  % number of replicates to perform at each step_size

% a series of matrices to store the L1 errors for each of the three
% optimization proceedures.
unweighted_errors = zeros(num_reps,length(support_sizes));
quikr_errors = zeros(num_reps,length(support_sizes));
weighted_errors = zeros(length(h_sizes), num_reps, length(support_sizes));
posteriori_test = zeros(length(h_sizes), num_reps, length(support_sizes));

%parpool()  % Either manually start the parallel pool, or else Matlab will
%do it automatically when it sees the parfor (unless you have disabled the
%option)

ppm = ParforProgMon('Please wait, working... ', length(support_sizes));

tic
parfor suppSizeInd=1:length(support_sizes)  % for each of the support sizes
    suppSize = support_sizes(suppSizeInd);
    
    % matrices to hold the errors at each replicated
    temp_l1 = zeros(1, num_reps);
    temp_q = zeros(1, num_reps);
    temp_weighteds = zeros(length(h_sizes), num_reps);
    temp_posteriori_test = zeros(length(h_sizes), num_reps);
    
    fprintf('On support size %d of %d\n', suppSizeInd, length(support_sizes));
    for rep=1:num_reps  % perform num_reps of replicates
        supp = datasample(1:num_species, suppSize, 'Replace', false);  % size of the support
        true_x = zeros(num_species,1);  % the true x vector we are trying to reconstruct
        true_x(supp) = rand(suppSize,1);
        true_x = true_x./sum(true_x);  % normalize to be a probability vector
        
        y_s = {}; % measurement vectors
        for i=1:length(h_sizes)
            y_s{i} = A_hs{i}*true_x;
        end
        
        % optimization parameters
        options = optimoptions('linprog','Algorithm','dual-simplex','Display','off','Diagnostics','off', 'ConstraintTolerance', .0000001, 'OptimalityTolerance', .0000001);

        % Unweighted L1 optimization
        [x_l1, ~, ~, ~, ~] = linprog_gurobi(ones(1,num_species), [], [], A_k_small, A_k_small*true_x, zeros(1,num_species), ones(1, num_species), options);
        %[x_l1, ~, ~, ~, ~] = linprog(ones(1,num_species), [], [],A_k_small, A_k_small*true_x, zeros(1,num_species), ones(1,num_species), options);  % if you do not have a Gurobi license, you can use the built in linprob
        x_l1 = x_l1./sum(x_l1);  % normalize
        temp_l1(rep) = sum(abs(true_x - x_l1));  % store the error
        
        % quikr
        lambda = 10000;
        tic;
        x_q = lsqnonneg([ones(1, num_species); lambda*A_k_small], [0;lambda*A_k_small*true_x]);
        x_q = x_q/sum(x_q);
        temp_q(rep) = sum(abs(true_x - x_q));

        % weighted optimization, over all h sizes
        for i=1:length(h_sizes)
            f = 1./(B_transpose_hs{i}*y_s{i}).^(1-q);
            [x_star, ~, ~, ~, ~] = linprog_gurobi(f, [], [], A_k_small, A_k_small*true_x, zeros(1,num_species), ones(1, num_species), options);  % again, can substitute linprog
            x_star = x_star./sum(x_star);
            temp_weighteds(i, rep) = sum(abs(true_x - x_star));
            temp_posteriori_test(i, rep) = sum(abs(A_k_large*true_x - A_k_large*x_star));
        end
    end
    unweighted_errors(:, suppSizeInd) = temp_l1;
    quikr_errors(:, suppSizeInd) = temp_q;
    weighted_errors(:, :, suppSizeInd) = temp_weighteds;
    posteriori_test(:, :, suppSizeInd) = temp_posteriori_test;
    ppm.increment()
end
fprintf('Finished\n')
toc

%% L1 norm error plot: plot of L1 error between true x and reconstructed x 
% (averaged over the number of replicates) as a function of support size ||x||_0
colors = linspecer(2+length(h_sizes));
fs = 15;
set(groot,'defaultAxesColorOrder', [0 0 0], 'DefaultAxesLineStyleOrder','-|--|:|-.|-*')
line_width = 2;
figure();
hold on
plot(support_sizes, mean(unweighted_errors), 'LineWidth', line_width, 'Color', colors(1,:))
plot(support_sizes, mean(quikr_errors), 'LineWidth', line_width, 'Color', colors(2,:))
for i=1:length(h_sizes)
     plot(support_sizes, mean(squeeze(weighted_errors(i,:,:))), 'LineWidth', line_width, 'MarkerSize', 4, 'Color', colors(i+2,:))
end
ylabel('Mean L1 error', 'FontSize', fs)
xlabel('Support size', 'FontSize', fs)
title(sprintf('Reconstruction performance: L1 norm using k = %d', small_k))
legends = {};
legends{1} = 'Unweighted';
legends{2} = 'Quikr';
for i=1:length(h_sizes)
       legends{i+2} = sprintf('Weighted, h = %d', h_sizes(i));
end
lgd = legend(legends{:});
lgd.FontSize = fs;
%set(gca,'DataAspectRatio',[15 1 1])
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
saveas(gcf, 'Figures/L1NormErrorMultiK.png')

%% Percent recovered plot:
% The percentage (over all replicates) of successful recoveries as a 
% function of the support size
thresh = 1e-3;  % anything L1 error smaller than this is considered "successful"

colors = linspecer(2+length(h_sizes));
fs = 15;
set(groot,'defaultAxesColorOrder', [0 0 0], 'DefaultAxesLineStyleOrder','-|--|:|-.|-*')
line_width = 2;
figure();
hold on
plot(support_sizes, mean(unweighted_errors<thresh), 'LineWidth', line_width, 'Color', colors(1,:))
plot(support_sizes, mean(quikr_errors<thresh), 'LineWidth', line_width, 'Color', colors(2,:))
for i=1:length(h_sizes)
     plot(support_sizes, mean(squeeze(weighted_errors(i,:,:))<thresh), 'LineWidth', line_width, 'MarkerSize', 4, 'Color', colors(i+2,:))
end
ylabel('Percent recovered', 'FontSize', fs)
xlabel('Support size', 'FontSize', fs)
legends = {};
legends{1} = '$\ell_1$';
legends{2} = 'Quikr';
for i=1:length(h_sizes)
       legends{i+2} = sprintf('h = %d', h_sizes(i));
end
lgd = legend(legends{:},'Interpreter','latex');
lgd.FontSize = fs;
set(gca,'DataAspectRatio',[20 1 1])
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
saveas(gcf, 'Figures/UnweightedVsWeightedMultiK.png')

%% Demonstrate the a posteriori test
thresh = 1e-5;  % anything L1 error smaller than this is considered "successful"

colors = linspecer(1+length(h_sizes));
fs = 15;
set(groot,'defaultAxesColorOrder', [0 0 0], 'DefaultAxesLineStyleOrder','-|--|:|-.|-*')
line_width = 2;
fig = figure();
set(fig,'defaultAxesColorOrder',[colors(1,:); colors(2,:)]);
hold on
i = length(h_sizes);
x = support_sizes;
yyaxis left
plot(x, mean(squeeze(weighted_errors(i,:,:))<thresh), 'LineWidth', line_width, 'MarkerSize', 4, 'Color', colors(1,:))
yyaxis right
plot(x, mean(squeeze(posteriori_test(i,:,:))), 'LineWidth', line_width, 'MarkerSize', 4, 'Color', colors(2,:))
yyaxis left
ylabel('Percent recovered', 'FontSize', fs)
xlabel('Support size', 'FontSize', fs)
yyaxis right
ylabel('$||A^{(h)}x - y^{(h)}||_1$', 'FontSize', fs, 'Interpreter','latex')
%set(fig,'DataAspectRatio',[112.5,1,2.5])
set(gca,'DataAspectRatio',[112.5,1,2.5])
set(gca,'PlotBoxAspectRatio',[1,0.533333333333333,0.533333333333333])
%x = 1.5;
%set(gcf,'Position',[x*100 x*100 x*500 x*500])
ax = gca;
%outerpos = ax.OuterPosition;
%ti = ax.TightInset; 
%left = outerpos(1) + ti(1);
%bottom = outerpos(2) + ti(2);
%ax_width = outerpos(3) - ti(1) - ti(3);
%ax_height = outerpos(4) - ti(2) - ti(4);
%ax.Position = [left bottom ax_width ax_height];
saveas(gcf, 'Figures/aposterioriMultiK.png')
%% export the data
if ~isfile('CooccurenceReproduciblesMultiData.mat')
    save('CooccurenceReproduciblesMultiData.mat')
end