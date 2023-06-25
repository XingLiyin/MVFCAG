clear all;
clc;
addpath datasets fun

%% load datasets
load('MSRC.mat');
num_x = size(Y,1); % number of samples
num_c = size(unique(Y),1); % number of classes
num_v = size(X,2); % number of views

%% Graph Construction
anchor = 0.1;
num_m = round(num_x*anchor);  % number of anchors
num_k = 5;  % number of k nearest neighbors

B = cell(num_v,1);
for i = 1:num_v
    % kmeans
    [B{i,1}] = GraphConstruction(X{i},num_m,num_k);
end

%% clustering
% choose the suitable value for lambda
step = 0.25;

% initialization
Y0 = randn(num_x,num_c);
Y0 = Y0 - min(Y0,[],2);
Y0 = projectm(Y0);
Z0 = cell(num_v,1);
for i = 1:num_v
    num_m = size(B{i,1},2);
    Z0{i} = randn(num_m,num_c);
    Z0{i} = Z0{i} - min(Z0{i},[],2);
    Z0{i} = projectm(Z0{i});
end
diff0 = -1;
pY0 = Y0;
pZ0 = Z0;
obj0 = zeros(201);
for lambda = 0.25:step:2.5
    [pY1,pZ1,obj1,diff1,regu1] = MVFCAG(B,Y0,Z0,lambda,num_c);
    if abs(diff0 - diff1) > 1
        diff = diff0;
        pY = pY0;
        pZ = pZ0;
        obj = obj0;
        diff0 = diff1;
        pY0 = pY1;
        pZ0 = pZ1;
        obj0 = obj1;
    else if diff1>1
            lambda = lambda - step;
            break;
        end
    end
end

% choose the best result 
% with the smallest value of objective function
objval0 = 10000;
for iter = 1:10
    % initialization
    Y0 = randn(num_x,num_c);
    Y0 = Y0 - min(Y0,[],2);
    Y0 = projectm(Y0);
    Z0 = cell(num_v,1);
    for i = 1:num_v
        num_m = size(B{i,1},2);
        Z0{i} = randn(num_m,num_c);
        Z0{i} = Z0{i} - min(Z0{i},[],2);
        Z0{i} = projectm(Z0{i});
    end
    % clustering
    tic;
    [pY,pZ,obj,diff,regu] = MVFCAG(B,Y0,Z0,lambda,num_c);
    timer = toc;
    objval = diff + regu;
    if objval<objval0
        objval0 = objval;
        pY0 = pY;
        pZ0 = pZ;
        obj0 = obj;
    end
end


[~,predY] = max(pY0,[],2);
result = ClusteringMeasure(Y, predY);



function x = projectm(m)
alpha0 = max(m,[],2);
alpha1 = min(m,[],2); %initialize
while any(abs(alpha1-alpha0) > 1e-4)
    alpha0 = alpha1; % save
    f = sum((m - alpha0).*(m>alpha0),2)-1;
    df = -sum((m>alpha0),2);
    alpha1 = alpha0 - f./df; % update
end
x = m - alpha1;
x(x<0) = 0;
end
