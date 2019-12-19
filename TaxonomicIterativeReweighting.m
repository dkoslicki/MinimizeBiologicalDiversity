%% This script implements the iterative reweighting scheme when utilizing a taxonomic similarity matrix
% Note: this works in MATLAB R2019a, but does not appear to work in version R2019b due to linprob_gurobi not playing well with R2019b

% basic imports
k = 4; %k-mer size
cols_vs_rows = 3;  % fix 3-times more columns than rows
num_species = cols_vs_rows*4^k;  % reduce the number of columns of the sensing matrix so pictures will be generated in a reasonable amount of time.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import pre-computed data
% The following series of computations took approx 80 minutes on an 18 core
% i9-9980XE processor. As such, you may wish to import the following data
% and skip directly to the visualizations
if isfile('TaxonomicIterativeReweightingData.mat')
    load('TaxonomicIterativeReweightingData.mat')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% import the data
addpath(genpath('Data'))
file = sprintf('97_otus_subset.fasta_A_%d_no_c.mat', k);
A_k = load(file);

T = load('97_otus_subset_genus_transfer_matrix.mat'); % this is the genus transfer matrix T from the paper
T = T.T;
large_group_inds = find(T*ones(size(T,2),1)>=5);  % genera with many representative species
% would like to select the columns such that not too many genera show up
% (i.e. have multiple organisms per genera)

has_genus = load('97_otus_subset_has_genera.mat');  % vector indicating which genera have representatives
has_genus = has_genus.has_genera;
genus_pos = find(has_genus);


%% select at least 5 organisms from each genus
to_select = [];
for large_group_ind_it = 1:length(large_group_inds)
    large_group_ind = large_group_inds(large_group_ind_it);
    genus_row = T(large_group_ind,:);
    select_count = 0;
    find_genus_row = find(genus_row);
    for organism_ind_it = 1:length(find_genus_row)
        organism_ind = find_genus_row(organism_ind_it);
        if has_genus(organism_ind)
            to_select = [to_select organism_ind];
            select_count = select_count + 1;
            if select_count >= 5
                break
            end
        end
    end
end
%% Reduce the problem down to the selected organisms
A_k = full(A_k.A_k(:, to_select(1:num_species)));  % sensing matrix
T = full(T(:, to_select(1:num_species)));
num_genera = length(find(sum(T,2)));


%% Simulations
% random vectors will be generated with a specified support size, and each
% of three methods will be used in an attempt to recover these vectors
% using equation (26)
% WARNING: this calculation takes a significant amount of time when
% small_k >= 4 (on the order of hours)

q = .01;  % fixed, small q value s.t. 0<q<1
my_eps = 1e-5;  % the epsilon in (26) itself
iter_thresh = 1e-3;  % the threshold by which to terminate the loop in iterative reweighting based on |x^{(n)} - x^{(n+1)}|
max_its = 25;
start = 70;  % starting support size
step_size = 2;  % how much to increase the support size in each step
max_support = 150;  % maximum support size
support_sizes = start:step_size:max_support;  % vector of support sizes
num_reps = 200;  % number of replicates to perform at each step_size

% a series of matrices to store the L1 errors for each of the three optimization proceedures.
errors = zeros(num_reps, length(support_sizes));  % iterative reweighting
times = zeros(num_reps, length(support_sizes));
errors_f = zeros(num_reps, length(support_sizes));  % feasibility test
times_f = zeros(num_reps, length(support_sizes));
errors_q = zeros(num_reps, length(support_sizes));  % quikr
times_q = zeros(num_reps, length(support_sizes));
its = zeros(num_reps, length(support_sizes));
options = optimoptions('linprog','Algorithm','dual-simplex','Display','off','Diagnostics','off', 'ConstraintTolerance', .000001, 'OptimalityTolerance', .000001);
%parpool()
%ppm = ParforProgMon('Please wait, working... ', length(support_sizes));

tic
parfor suppSizeInd=1:length(support_sizes)  % for each of the support sizes
    suppSize = support_sizes(suppSizeInd);
    temp_l1 = zeros(1, num_reps);
    temp_l1_time = zeros(1, num_reps);
    temp_l1_q = zeros(1, num_reps);
    temp_l1_q_time = zeros(1, num_reps);
    temp_l1_f = zeros(1, num_reps);
    temp_l1_f_time = zeros(1, num_reps);
    temp_its = zeros(1, num_reps);
    fprintf('On support size %d of %d\n', suppSizeInd, length(support_sizes));
    for rep=1:num_reps  % perform num_reps of replicates
        supp = datasample(1:num_species, suppSize, 'Replace', false);  % size of the support
        true_x = zeros(num_species,1);  % the true x vector we are trying to reconstruct
        true_x(supp) = rand(suppSize,1);
        true_x = true_x./sum(true_x);  % normalize to be a probability vector
        y = A_k*true_x;
        
        % feasibility test
        tic;
        [x_f, ~, ~, ~, ~] = linprog_gurobi(zeros(size(A_k,2),1), [], [], A_k, y, zeros(1,num_species), ones(1, num_species), options);
        temp_l1_f_time(rep) = toc;
        
        % quikr
        lambda = 10000;
        tic;
        x_q = lsqnonneg([ones(1, size(A_k,2)); lambda*A_k], [0;lambda*y]);
        temp_l1_q_time(rep) = toc;
        x_q = x_q/sum(x_q);
        
        % Iterative reweighting L1 optimization
        x_np1 = zeros(num_species, 1);
        it = 0;
        x_n = x_f; % initial feasible vector. 
        tic;
        while it < max_its
            f = 1./(sum(T.*(T*x_n)) + my_eps).^(1-q);
            [x_np1, ~, ~, ~, ~] = linprog_gurobi(f, [], [], A_k, y, zeros(1,num_species), ones(1, num_species), options);
            x_np1 = x_np1./sum(x_np1);
            
            if norm(x_n - x_np1,1) < iter_thresh
                break
            else
                x_n = x_np1;
            end
            it = it + 1;
        end
        temp_l1_time(rep) = toc;
        
        temp_l1(rep) = sum(abs(T*true_x - T*x_np1));  % genus level error for iterative procedure
        temp_l1_f(rep) = sum(abs(T*true_x - T*x_f));  % genus level error for feasibility test
        temp_l1_q(rep) = sum(abs(T*true_x - T*x_q));  % genus level error for Quikr
        temp_its(rep) = it;

    end
    errors(:, suppSizeInd) = temp_l1;
    errors_f(:, suppSizeInd) = temp_l1_f;
    errors_q(:, suppSizeInd) = temp_l1_q;
    times(:, suppSizeInd) = temp_l1_time;
    times_f(:, suppSizeInd) = temp_l1_f_time;
    times_q(:, suppSizeInd) = temp_l1_q_time;
    its(:, suppSizeInd) = temp_its;
    %ppm.increment()
end
fprintf('Finished\n')
toc

%% Vizualization: L1 norm error plot: plot of L1 error between true x and reconstructed x 
% (averaged over the number of replicates) as a function of support size ||x||_0
colors = linspecer(2);
fs = 15;
set(groot,'defaultAxesColorOrder', [0 0 0], 'DefaultAxesLineStyleOrder','-|--|:|-.|-*')
line_width = 2;
figure();
hold on
plot(start:step_size:max_support, mean(errors), 'LineWidth', line_width, 'Color', colors(1,:))
plot(start:step_size:max_support, mean(errors_f), 'LineWidth', line_width, 'Color', 'r')
plot(start:step_size:max_support, mean(errors_q), 'LineWidth', line_width, 'Color', 'k')
ylabel('Mean L1 error', 'FontSize', fs)
xlabel('Support size', 'FontSize', fs)
title(sprintf('Reconstruction performance: L1 norm using k = %d', k))
lgd = legend('IRWLP', 'Feasibility', 'Quikr');
lgd.FontSize = fs;
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf, 'Figures/ItDecreaseDivQuikrFeasibleL1Error.png')

%% Vizualization: Percent recovered plot:
% The percentage (over all replicates) of successful recoveries as a 
% function of the support size
thresh = 1e-3;  % anything L1 error smaller than this is considered "successful"

colors = linspecer(2);
fs = 15;
set(groot,'defaultAxesColorOrder', [0 0 0], 'DefaultAxesLineStyleOrder','-|--|:|-.|-*')
line_width = 2;
figure();
hold on
plot(support_sizes, mean(errors<thresh), 'LineWidth', line_width, 'Color', colors(1,:))
plot(support_sizes, mean(errors_f<thresh), 'LineWidth', line_width, 'Color', 'r')
plot(support_sizes, mean(errors_q<thresh), 'LineWidth', line_width, 'Color', 'k')
ylabel('Percent recovered', 'FontSize', fs)
xlabel('Support size', 'FontSize', fs)
lgd = legend('(IRWLP)', 'Feasibility', 'Quikr');
lgd.FontSize = fs;
%set(gca,'DataAspectRatio',[80,1,2])
%set(gca,'PlotBoxAspectRatio',[1,0.496900826446281,0.496900826446281])
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf, 'Figures/ItDecreaseDivQuikrFeasible.png')
%% Vizualization: Timing plot
colors = linspecer(2);
fs = 15;
trunc_val = 119;  % where to truncate the x-axis
trunc = find(support_sizes>trunc_val);
trunc = trunc(1);
set(groot,'defaultAxesColorOrder', [0 0 0], 'DefaultAxesLineStyleOrder','-|--|:|-.|-*')
line_width = 2;
figure();
hold on
plot(support_sizes(1:trunc), mean(times(:, 1:trunc)), 'LineWidth', line_width, 'Color', colors(1,:))
plot(support_sizes(1:trunc), mean(times_f(:, 1:trunc)), 'LineWidth', line_width, 'Color', 'r')
plot(support_sizes(1:trunc), mean(times_q(:, 1:trunc)), 'LineWidth', line_width, 'Color', 'k')
ylabel('Seconds', 'FontSize', fs)
xlabel('Support size', 'FontSize', fs)
lgd = legend('(IRWLP)', 'Feasibility', 'Quikr');
lgd.FontSize = fs;
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset; 
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
saveas(gcf, 'Figures/ItDecreaseDivQuikrFeasibleTime.png')
%% 
% and it appears that the iterations are actually doing something since 
% there exists times when its>1 and the reconstruction was successfull (so
% it wasn't a simple feasibility problem).
length(find(and(errors<thresh, its>1)))  % Number of times the recovery was successful and the number of iterates of the reweighting was 2 or higher

%% export data
if ~isfile('TaxonomicIterativeReweightingData.mat')
    save('TaxonomicIterativeReweightingData.mat')
end