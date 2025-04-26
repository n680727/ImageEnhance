% 清除工作空間和命令視窗
clear all;
clc;

% 檢查 GPU 是否可用
if isempty(gpuDevice)
    warning('未檢測到 GPU，程式將在 CPU 上執行。');
else
    disp(['正在使用 GPU: ', gpuDevice().Name]);
end

% 指定要處理的色彩空間（在這裡修改）
colorSpace = 'HSV'; % 可選：'HSV', 'LAB', 'YCBCR'

% 驗證色彩空間選擇
validColorSpaces = {'HSV', 'LAB', 'YCBCR'};
if ~ismember(upper(colorSpace), validColorSpaces)
    error('無效的色彩空間！請選擇 ''HSV'', ''LAB'' 或 ''YCBCR''。');
end
colorSpace = upper(colorSpace); % 轉為大寫以保持一致

% 定義資料集路徑和輸出路徑
datasetPath = 'D:\program\school\ImageEnhance\dataset\darkface1000'; % 資料集路徑
outputBasePath = 'D:\program\school\ImageEnhance\dataset\MPA'; % 輸出路徑

% 確保輸出基礎資料夾存在
if ~exist(outputBasePath, 'dir')
    mkdir(outputBasePath);
end

% 獲取資料夾中所有 .png 檔案
imageFiles = dir(fullfile(datasetPath, '*.png')); % 列出所有 .png 檔案
numImages = min(1, length(imageFiles)); % 先測試最多 300 張
%numImages = length(imageFiles); % 圖片總數

% 提取檔案名稱中的編號並進行數值排序
[~, fileNames, ~] = cellfun(@fileparts, {imageFiles.name}, 'UniformOutput', false);
fileNumbers = str2double(fileNames); % 將檔案名稱轉為數值（例如 '18' -> 18）
[~, sortIdx] = sort(fileNumbers); % 按照數值排序
imageFiles = imageFiles(sortIdx); % 按數值順序重新排列檔案

% 初始化用於儲存所有圖片的 PCQI 分數
all_pcqi_scores = zeros(numImages, 3); % 處理後圖片：公式 1, 2, 3
all_pcqi_scores_blended = zeros(numImages, 3); % 混合圖片：公式 1, 2, 3

% 定義混合權重 alpha (範圍 0 到 1)
alpha = 0.5; % 你可以調整這個值，例如 0.3 表示原始圖片佔 30%，處理後圖片佔 70%

% 確保 alpha 在有效範圍內
if alpha < 0 || alpha > 1
    error('alpha 必須在 0 到 1 之間！');
end

% 迴圈處理每張圖片
for imgIdx = 1:numImages
    % 顯示當前處理進度
    disp(sprintf('正在處理第 %d/%d 張圖片', imgIdx, numImages));
    
    % 獲取當前圖片的名稱和完整路徑
    imageName = fullfile(datasetPath, imageFiles(imgIdx).name);
    
    % 讀取圖片
    inputImage = imread(imageName);
    
    % 檢查圖片是否為 RGB 格式
    if size(inputImage, 3) ~= 3
        disp(['圖片 ', imageName, ' 不是 RGB 格式，跳過此圖片。']);
        continue;
    end
    
    % 提取圖片檔名和副檔名
    [~, baseName, ext] = fileparts(imageName); % 例如 '18.png' -> baseName='18', ext='.png'
    
    % 為每張圖片創建子資料夾
    parentFolder = fullfile(outputBasePath, baseName);
    if ~exist(parentFolder, 'dir')
        mkdir(parentFolder);
    end
    subFolderPath = fullfile(parentFolder, colorSpace);
    if ~exist(subFolderPath, 'dir')
        mkdir(subFolderPath);
    end
    
    % 根據輸入圖片的副檔名決定輸出副檔名
    outputExt = ext; % 輸出格式與輸入格式相同（例如 .png）
    
    % 將圖片像素值正規化到 [0, 1] 範圍，並轉移到 GPU
    inputImageNorm = gpuArray(double(inputImage) / 255);
    
    % 儲存原始圖片的灰階版本，用於 PCQI 評估
    inputGray = gpuArray(double(rgb2gray(inputImage)));
    
    % --- MPA 優化參數 a 和 b（針對公式 3） ---
    % MPA 參數設置
    nPop = 30; % 掠食者數量
    Max_iter = 15; % 迭代次數
    dim = 2; % 優化變數數量 (a 和 b)
    lb = [1, 0.5]; % 下界 (a: 1, b: 0.01)
    ub = [100, 3]; % 上界 (a: 100, b: 3)
    P = 0.5; % 步長控制參數
    FADS = 0.1; % 渦流效應機率
    
    % 初始化掠食者與獵物位置（在 GPU 上）
    Prey = gpuArray(zeros(nPop, dim));
    for i = 1:nPop
        Prey(i, :) = lb + (ub - lb) .* rand(1, dim); % 隨機初始化獵物
    end
    Predator = Prey; % 初始時掠食者與獵物位置相同
    
    % 初始化 Elite Matrix（儲存最佳掠食者）
    Elite = gpuArray(zeros(1, dim)); % 最佳掠食者位置
    Elite_fitness = -inf; % 最佳 PCQI 分數（最大化問題）
    
    % 迭代優化
    Convergence_curve = zeros(1, Max_iter);
    for iter = 1:Max_iter
        % 計算當前掠食者的適應度
        for i = 1:nPop
            % 確保參數在範圍內
            Prey(i, :) = max(Prey(i, :), lb);
            Prey(i, :) = min(Prey(i, :), ub);
            Predator(i, :) = max(Predator(i, :), lb);
            Predator(i, :) = min(Predator(i, :), ub);
            
            % 提取當前掠食者的 a 和 b
            a = Prey(i, 1);
            b = Prey(i, 2);
            
            % 根據色彩空間處理圖片（使用公式 3 計算 PCQI）
            switch colorSpace
                case 'HSV'
                    hsvImage = rgb2hsv(inputImageNorm);
                    x = hsvImage(:,:,3);
                    y3 = log(1 + a * (x.^b)) / log(1 + a); % 公式 3
                    y3 = max(0, min(1, y3));
                    hsvImage3 = hsvImage; hsvImage3(:,:,3) = y3;
                    outputImage3 = uint8(gather(hsv2rgb(hsvImage3) * 255)); % 從 GPU 取回數據
                case 'LAB'
                    labImage = rgb2lab(inputImageNorm);
                    x = labImage(:,:,1) / 100;
                    y3 = log(1 + a * (x.^b)) / log(1 + a); % 公式 3
                    y3 = max(0, min(1, y3));
                    labImage3 = labImage; labImage3(:,:,1) = y3 * 100;
                    outputImage3 = uint8(gather(lab2rgb(labImage3) * 255));
                case 'YCBCR'
                    ycbcrImage = rgb2ycbcr(inputImageNorm);
                    x = ycbcrImage(:,:,1);
                    y3 = log(1 + a * (x.^b)) / log(1 + a); % 公式 3
                    y3 = max(0, min(1, y3));
                    ycbcrImage3 = ycbcrImage; ycbcrImage3(:,:,1) = y3;
                    outputImage3 = uint8(gather(ycbcr2rgb(ycbcrImage3) * 255));
            end
            
            % 計算 PCQI 分數
            outputGray3 = gpuArray(double(rgb2gray(outputImage3)));
            [fitness, ~] = PCQI(gather(inputGray), gather(outputGray3)); % PCQI 可能需要 CPU 數據
            fitness = double(fitness);
            
            % 更新 Elite Matrix
            if fitness > Elite_fitness
                Elite_fitness = fitness;
                Elite = Prey(i, :);
            end
        end
        
        % MPA 三階段更新
        current_iter = iter / Max_iter; % 正規化迭代進度
        RB = rand(nPop, dim); % 布朗運動隨機向量
        RL = 0.5 * (rand(nPop, dim) - 0.5); % 列維飛行隨機向量
        stepsize = gpuArray(zeros(nPop, dim));
        
        % 適應因子 CF
        CF = (1 - iter / Max_iter)^(2 * iter / Max_iter);
        
        for i = 1:nPop
            for j = 1:dim
                R = rand(); % 隨機數
                if current_iter < 1/3 % 第一階段：高速度比（探索）
                    stepsize(i, j) = RB(i, j) * (Elite(j) - RB(i, j) * Prey(i, j));
                    Predator(i, j) = Prey(i, j) + P * stepsize(i, j);
                elseif current_iter >= 1/3 && current_iter < 2/3 % 第二階段：單位速度比（過渡）
                    if i <= nPop / 2
                        stepsize(i, j) = RB(i, j) * (Elite(j) - RB(i, j) * Prey(i, j));
                        Predator(i, j) = Prey(i, j) + P * stepsize(i, j);
                    else
                        stepsize(i, j) = RL(i, j) * (RL(i, j) * Elite(j) - Prey(i, j));
                        Predator(i, j) = Elite(j) + P * CF * stepsize(i, j);
                    end
                else % 第三階段：低速度比（開發）
                    stepsize(i, j) = RL(i, j) * (RL(i, j) * Elite(j) - Prey(i, j));
                    Predator(i, j) = Elite(j) + P * CF * stepsize(i, j);
                end
                % 渦流效應（FADS）
                if rand() < FADS
                    U = rand() < 0.5;
                    r = rand();
                    if U
                        Predator(i, j) = lb(j) + r * (ub(j) - lb(j));
                    else
                        r1 = rand();
                        r2 = rand();
                        Predator(i, j) = Predator(i, j) + (r1 * (ub(j) - lb(j)) * r2);
                    end
                end
            end
        end
        
        % 更新獵物位置
        Prey = Predator;
        
        % 儲存當前迭代的最佳分數
        Convergence_curve(iter) = Elite_fitness;
        
        % 顯示當前迭代資訊
        disp(['圖片 ', baseName, ' - 迭代 ', num2str(iter), ': a = ', num2str(gather(Elite(1))), ...
              ', b = ', num2str(gather(Elite(2))), ', PCQI = ', num2str(Elite_fitness)]);
    end
    
    % 最佳參數
    best_a = gather(Elite(1));
    best_b = gather(Elite(2));
    best_pcqi = Elite_fitness;
    
    % 顯示最佳參數和 PCQI 分數
    disp(['圖片 ', baseName, ' - MPA 優化結果（公式 3）：']);
    disp(['最佳 a: ', num2str(best_a)]);
    disp(['最佳 b: ', num2str(best_b)]);
    disp(['最佳 PCQI 分數 (公式 3): ', num2str(best_pcqi)]);
    
    % 繪製收斂曲線
    figure('Visible', 'off'); % 不顯示圖形視窗
    plot(1:Max_iter, Convergence_curve, 'LineWidth', 2);
    xlabel('迭代次數');
    ylabel('PCQI 分數');
    title(['圖片 ', baseName, ' - MPA 收斂曲線 (公式 3)']);
    grid on;
    
    % 儲存收斂曲線圖
    convergenceFileName = fullfile(parentFolder, colorSpace, [colorSpace, '_mpa_convergence', outputExt]);
    saveas(gcf, convergenceFileName);
    disp(['圖片 ', baseName, ' - MPA 收斂曲線已儲存為 ', convergenceFileName]);
    close(gcf); % 關閉圖形視窗
    
    % 使用最佳參數重新處理圖片
    a = best_a;
    b = best_b;
    
    % 初始化用於儲存 PCQI 分數的變數
    pcqi_scores = zeros(1, 3); % 1 種色彩空間 x 3 個公式（處理後圖片）
    pcqi_scores_blended = zeros(1, 3); % 1 種色彩空間 x 3 個公式（混合圖片）
    
    % 根據選擇的色彩空間進行處理
    switch colorSpace
        case 'HSV'
            % 將 RGB 轉換為 HSV
            hsvImage = rgb2hsv(inputImageNorm);
            x = hsvImage(:,:,3); % 提取 V 通道（亮度）
            
            % 公式 1: y = log(1 + a * x) / log(1 + a)
            y1 = log(1 + a * x) / log(1 + a);
            y1 = max(0, min(1, y1));
            
            % 公式 2: y = x^b
            y2 = x.^b;
            y2 = max(0, min(1, y2));
            
            % 公式 3: y = log(1 + a * x^b) / log(1 + a)
            y3 = log(1 + a * (x.^b)) / log(1 + a);
            y3 = max(0, min(1, y3));
            
            % 將處理後的 V 通道放回 HSV 圖像
            hsvImage1 = hsvImage; hsvImage1(:,:,3) = y1;
            hsvImage2 = hsvImage; hsvImage2(:,:,3) = y2;
            hsvImage3 = hsvImage; hsvImage3(:,:,3) = y3;
            
            % 將 HSV 轉回 RGB
            outputImage1 = uint8(gather(hsv2rgb(hsvImage1) * 255));
            outputImage2 = uint8(gather(hsv2rgb(hsvImage2) * 255));
            outputImage3 = uint8(gather(hsv2rgb(hsvImage3) * 255));
            
        case 'LAB'
            % 將 RGB 轉換為 LAB
            labImage = rgb2lab(inputImageNorm);
            x = labImage(:,:,1) / 100; % 提取 L 通道（亮度）並正規化到 [0, 1]
            
            % 公式 1: y = log(1 + a * x) / log(1 + a)
            y1 = log(1 + a * x) / log(1 + a);
            y1 = max(0, min(1, y1));
            
            % 公式 2: y = x^b
            y2 = x.^b;
            y2 = max(0, min(1, y2));
            
            % 公式 3: y = log(1 + a * x^b) / log(1 + a)
            y3 = log(1 + a * (x.^b)) / log(1 + a);
            y3 = max(0, min(1, y3));
            
            % 將處理後的 L 通道放回 LAB 圖像
            labImage1 = labImage; labImage1(:,:,1) = y1 * 100;
            labImage2 = labImage; labImage2(:,:,1) = y2 * 100;
            labImage3 = labImage; labImage3(:,:,1) = y3 * 100;
            
            % 將 LAB 轉回 RGB
            outputImage1 = uint8(gather(lab2rgb(labImage1) * 255));
            outputImage2 = uint8(gather(lab2rgb(labImage2) * 255));
            outputImage3 = uint8(gather(lab2rgb(labImage3) * 255));
            
        case 'YCBCR'
            % 將 RGB 轉換為 YCbCr
            ycbcrImage = rgb2ycbcr(inputImageNorm);
            x = ycbcrImage(:,:,1); % 提取 Y 通道（亮度）
            
            % 公式 1: y = log(1 + a * x) / log(1 + a)
            y1 = log(1 + a * x) / log(1 + a);
            y1 = max(0, min(1, y1));
            
            % 公式 2: y = x^b
            y2 = x.^b;
            y2 = max(0, min(1, y2));
            
            % 公式 3: y = log(1 + a * x^b) / log(1 + a)
            y3 = log(1 + a * (x.^b)) / log(1 + a);
            y3 = max(0, min(1, y3));
            
            % 將處理後的 Y 通道放回 YCbCr 圖像
            ycbcrImage1 = ycbcrImage; ycbcrImage1(:,:,1) = y1;
            ycbcrImage2 = ycbcrImage; ycbcrImage2(:,:,1) = y2;
            ycbcrImage3 = ycbcrImage; ycbcrImage3(:,:,1) = y3;
            
            % 將 YCbCr 轉回 RGB
            outputImage1 = uint8(gather(ycbcr2rgb(ycbcrImage1) * 255));
            outputImage2 = uint8(gather(ycbcr2rgb(ycbcrImage2) * 255));
            outputImage3 = uint8(gather(ycbcr2rgb(ycbcrImage3) * 255));
            
        otherwise
            error('程式碼錯誤：未正確設置色彩空間！');
    end
    
    % 儲存處理後圖片到對應子資料夾
    imwrite(outputImage1, fullfile(parentFolder, colorSpace, [colorSpace, '_formula1', outputExt]));
    imwrite(outputImage2, fullfile(parentFolder, colorSpace, [colorSpace, '_formula2', outputExt]));
    imwrite(outputImage3, fullfile(parentFolder, colorSpace, [colorSpace, '_formula3', outputExt]));
    
    % 混合圖片
    inputImageGPU = gpuArray(double(inputImage));
    blendedImage1 = uint8(gather(alpha * inputImageGPU + (1 - alpha) * gpuArray(double(outputImage1))));
    blendedImage2 = uint8(gather(alpha * inputImageGPU + (1 - alpha) * gpuArray(double(outputImage2))));
    blendedImage3 = uint8(gather(alpha * inputImageGPU + (1 - alpha) * gpuArray(double(outputImage3))));
    
    % 儲存混合圖片到對應子資料夾
    imwrite(blendedImage1, fullfile(parentFolder, colorSpace, [colorSpace, '_formula_mix1', outputExt]));
    imwrite(blendedImage2, fullfile(parentFolder, colorSpace, [colorSpace, '_formula_mix2', outputExt]));
    imwrite(blendedImage3, fullfile(parentFolder, colorSpace, [colorSpace, '_formula_mix3', outputExt]));
    
    % 計算 PCQI（處理後圖片）
    outputGray1 = double(rgb2gray(outputImage1));
    outputGray2 = double(rgb2gray(outputImage2));
    outputGray3 = double(rgb2gray(outputImage3));
    [mpcqi1, ~] = PCQI(gather(inputGray), outputGray1);
    [mpcqi2, ~] = PCQI(gather(inputGray), outputGray2);
    [mpcqi3, ~] = PCQI(gather(inputGray), outputGray3);
    pcqi_scores(1, :) = [mpcqi1, mpcqi2, mpcqi3];
    
    % 計算 PCQI（混合圖片）
    blendedGray1 = double(rgb2gray(blendedImage1));
    blendedGray2 = double(rgb2gray(blendedImage2));
    blendedGray3 = double(rgb2gray(blendedImage3));
    [mpcqi1_blended, ~] = PCQI(gather(inputGray), blendedGray1);
    [mpcqi2_blended, ~] = PCQI(gather(inputGray), blendedGray2);
    [mpcqi3_blended, ~] = PCQI(gather(inputGray), blendedGray3);
    pcqi_scores_blended(1, :) = [mpcqi1_blended, mpcqi2_blended, mpcqi3_blended];
    
    % 儲存 PCQI 分數
    all_pcqi_scores(imgIdx, :) = pcqi_scores;
    all_pcqi_scores_blended(imgIdx, :) = pcqi_scores_blended;
    
    % 顯示 PCQI 分數（處理後圖片和混合圖片）
    disp(['圖片 ', baseName, ' - PCQI 分數（處理後圖片） - ', colorSpace, ' 色彩空間：']);
    disp(['公式 1: ', num2str(mpcqi1)]);
    disp(['公式 2: ', num2str(mpcqi2)]);
    disp(['公式 3: ', num2str(mpcqi3)]);
    
    disp(['圖片 ', baseName, ' - PCQI 分數（混合圖片） - ', colorSpace, ' 色彩空間：']);
    disp(['公式 1: ', num2str(mpcqi1_blended)]);
    disp(['公式 2: ', num2str(mpcqi2_blended)]);
    disp(['公式 3: ', num2str(mpcqi3_blended)]);
    
    % 顯示儲存完成訊息
    disp(['圖片 ', baseName, ' - 處理完成：']);
    disp([[colorSpace, ' 處理圖片已儲存為 '], fullfile(parentFolder, colorSpace, [colorSpace, '_formula1', outputExt]), ', ', ...
          fullfile(parentFolder, colorSpace, [colorSpace, '_formula2', outputExt]), ', ', ...
          fullfile(parentFolder, colorSpace, [colorSpace, '_formula3', outputExt])]);
    disp([[colorSpace, ' 混合圖片已儲存為 '], fullfile(parentFolder, colorSpace, [colorSpace, '_formula_mix1', outputExt]), ', ', ...
          fullfile(parentFolder, colorSpace, [colorSpace, '_formula_mix2', outputExt]), ', ', ...
          fullfile(parentFolder, colorSpace, [colorSpace, '_formula_mix3', outputExt])]);
end

% 計算 PCQI 平均值（僅考慮成功處理的圖片）
valid_indices = find(all_pcqi_scores(:, 1) ~= 0); % 找到非零的 PCQI 分數（即成功處理的圖片）
if ~isempty(valid_indices)
    avg_pcqi_scores = mean(all_pcqi_scores(valid_indices, :), 1);
    avg_pcqi_scores_blended = mean(all_pcqi_scores_blended(valid_indices, :), 1);
    
    % 顯示 300 張圖片的 PCQI 平均值
    disp(['所有圖片的 PCQI 平均值（處理後圖片）：']);
    disp(['公式 1: ', num2str(avg_pcqi_scores(1))]);
    disp(['公式 2: ', num2str(avg_pcqi_scores(2))]);
    disp(['公式 3: ', num2str(avg_pcqi_scores(3))]);
    
    disp(['所有張圖片的 PCQI 平均值（混合圖片）：']);
    disp(['公式 1: ', num2str(avg_pcqi_scores_blended(1))]);
    disp(['公式 2: ', num2str(avg_pcqi_scores_blended(2))]);
    disp(['公式 3: ', num2str(avg_pcqi_scores_blended(3))]);
    
    % 顯示成功處理的圖片數量
    disp(['成功處理的圖片數量：', num2str(length(valid_indices)), ' / ', num2str(numImages)]);
else
    disp('沒有成功處理的圖片，無法計算 PCQI 平均值。');
end