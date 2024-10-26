function [solution, time, off, ofit, site, paretoAVE, tempVar, bitImportance] = SparseEA(features, labels, lower, upper, variableTypes, N, cnti)
    fprintf('SparseEA\n');                                      
    tic
    global maxFES
    global sizep
    global CNTTIME
    FES = 1;
    dim = size(features, 2);
    ofit = zeros(N, 2); % Objective function values for the population
    initThres = 1;
    thres = 0.1; % Exponential decay constant
    paretoAVE = zeros(1, 2); % To save final result of the Pareto front
    
    %% Initialization
    TDec = []; 
    TMask = [];
    TempPop = [];
    dimFitness = zeros(1, dim);
    Tobj = zeros(0, 2); % Initialize Tobj as an empty matrix or zero matrix with appropriate size
    
    for i = 1 : 1 + 4 * any(variableTypes ~= 4)
        Dec = unifrnd(repmat(lower, dim, 1), repmat(upper, dim, 1));
        Dec(:, variableTypes == 4) = 1;
        Mask = eye(dim);
        pop = Dec .* Mask;
        TDec = [TDec; Dec];
        TMask = [TMask; Mask];
        TempPop = [TempPop; pop];
        dimfit = zeros(dim, 2);
        for m = 1 : dim
            [dimfit(m, 1), dimfit(m, 2)] = FSKNNfeixiang(pop(m, :), features, labels);
        end
        Tobj = [Tobj; dimfit]; % Store the objective function values
        dimFitness = dimFitness + NDSort(dimfit, dim);  % The order of the Pareto front is used as Fitness
    end
    
    % Generate initial population
    Dec = unifrnd(repmat(lower, N, 1), repmat(upper, N, 1));
    Dec(:, variableTypes == 4) = 1;
    Mask = false(N, dim);
    for i = 1 : N
        Mask(i, TournamentSelection(2, ceil(rand * dim), dimFitness)) = 1;
    end
    off = logical(Dec .* Mask);
    
    %% Evaluate
    for i = 1 : N
        [ofit(i, 1), ofit(i, 2)] = FSKNNfeixiang(off(i, :), features, labels);
    end
    [FrontNO, ~] = NDSort(ofit(:, 1:2), N);
    site = find(FrontNO == 1);
    solution = ofit(site, :);
    solution(:, 2) = solution(:, 2) / dim;
    disp('Solution:');
    disp(solution);
    erBestParetoAVE = 1;  % To save the history best
    paretoAVE(1) = mean(solution(:, 1));
    paretoAVE(2) = mean(solution(:, 2));
    
    %% MI
    Y_train = labels;
    MI = zeros(1, dim);
    for i = 1 : dim
        MI(i) = MItest(features(:, i), Y_train);
    end
    
    %% DR
    DR = zeros(1, N);
    for i = 1 : N
        DR(i) = sum(FrontNO > FrontNO(i));
    end
    DR = DR ./ N;
    
    %% Initialization bitImportance
    bitImportance = zeros(N, dim);
    for i = 1 : N
        for j = 1 : dim
            bitImportance(i, j) = MI(j) * DR(i) / (sum(MI) * N);
        end
    end
    
    %% 
    tAveError = zeros(1, maxFES);  % All error rates
    tAveFea = zeros(1, maxFES);   % All feature numbers
    tErBest = zeros(1, maxFES);
    tThres = zeros(1, maxFES);
    tempVar = cell(1, 4);
    
    %% Main loop
    while FES <= maxFES
        isChange = zeros(N, dim); 
        extTemp = 0; 
        
        %---------------- Dimensionality reduction ---------------
        for i = 1 : N
            if ismember(i, site)
                continue;
            end
            
            curiOff = off(i, :); 
            curpSite = site(randi(length(site))); 
            pop = off(curpSite, :); 
            
            aveiBit = mean(bitImportance(i, :));
            
            for j = 1 : dim
                popBit = pop(j);
                ext = 1 / (1 + exp(-5 * (aveiBit - bitImportance(i, j))));
                tempThres = initThres * exp(-thres * FES);
                ext = ext * tempThres;
                if rand() < ext
                    off(i, j) = 0; 
                end
                extTemp = extTemp + ext;
                
                %----- Individual repairing -----
                if bitImportance(i, j) > bitImportance(curpSite, j)
                    off(i, j) = curiOff(j); 
                else
                    if rand() < (bitImportance(curpSite, j) - bitImportance(i, j)) / bitImportance(curpSite, j)
                        off(i, j) = popBit;
                    end
                end
                
                if curiOff(j) ~= off(i, j)
                    isChange(i, j) = 1;
                end
            end         
        end
        extTemp = extTemp / dim / N;
        tThres(FES) = extTemp;
        
        %--------------- Evaluate ----------------
        for i = 1 : N
            [ofit(i, 1), ofit(i, 2)] = FSKNNfeixiang(off(i, :), features, labels);
        end
        [FrontNO, ~] = NDSort(ofit(:, 1:2), N);
        site = find(FrontNO == 1);
        solution = ofit(site, :);
        oldERAVE = paretoAVE(1);
        paretoAVE(1) = mean(solution(:, 1)); 
        paretoAVE(2) = mean(solution(:, 2));
        if paretoAVE(1) < erBestParetoAVE
            erBestParetoAVE = paretoAVE(1); 
        end
        oldDR = DR;
        for i = 1 : N
            DR(i) = sum(FrontNO > FrontNO(i));
        end
        DR = DR ./ N;
        
        %---- Update SM ----
        if paretoAVE(1) >= oldERAVE
            oldBI = bitImportance;
            for i = 1 : N
                for j = 1 : dim
                    bitImportance(i, j) = 0.7 * oldBI(i, j) + 0.3 * MI(j) * DR(i) / (sum(MI) * N);
                end
            end
        else
            for i = 1 : N
                for j = 1 : dim
                    tempIndex = site(randi(length(site)));
                    if (FrontNO(i) <= FrontNO(tempIndex))
                        if isChange(i, j) == 1 && off(i, j) == 1
                            bitImportance(i, j) = bitImportance(i, j) * exp(0.8) ^ (1 / sqrt(1 + 1.0));
                            bitImportance(i, j) = min(bitImportance(i, j), 1.0);
                        elseif isChange(i, j) == 1 && off(i, j) == 0
                            bitImportance(i, j) = bitImportance(i, j) * exp(-0.2) ^ (1 / sqrt(1 + 1.0));
                        end
                    else
                        %if DR(i) > oldDR(i)
                         %   bitImportance(i, j) = bitImportance(i, j) * exp(0.5) ^ (1 / sqrt(1 + 1.0));
                          %  bitImportance(i, j) = min(bitImportance(i, j), 1.0);
                       % else
                        %    bitImportance(i, j) = bitImportance(i, j) * exp(-0.5) ^ (1 / sqrt(1 + 1.0));
                      %  end
                    end
                end
            end
        end
        
        erBestID = find(ofit(:, 1) == min(ofit(:, 1)));
        erBestID = erBestID(1);
        TEMPCNT = ((cnti - 1) * maxFES + FES) / CNTTIME;
        fprintf('PRG: %.1f%%-- GEN: %2d  Error: %.5f  F: %.2f     ErBest: %.5f     thres: %.5f\n',100*TEMPCNT, FES,paretoAVE(1),paretoAVE(2),ofit(erBestID,1),extTemp);
        tAveError(FES) = paretoAVE(1);
        tAveFea(FES) = paretoAVE(2);
        tErBest(FES) = ofit(erBestID,1);
        FES = FES + 1;
    end
    %%
     
    [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
    site = find(FrontNO==1);
    solution = ofit(site,:);
    
    paretoAVE(1) = mean(solution(:,1));
    paretoAVE(2) = mean(solution(:,2));
    tempVar{1} = tAveError;
    tempVar{2} = tAveFea;
    tempVar{3} = tErBest;
    tempVar{4} = tThres;
    clear tAveError;
    clear tAveFea;
    clear tErBest;
    clear tThres;
    toc
    time = toc;
 end