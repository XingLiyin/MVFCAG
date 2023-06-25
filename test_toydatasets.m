clear all;
clc;
addpath datasets records fun
%% generate datasets
% generate twomoon dataset
num_x = 300;
num_c = 2;
num_v = 3;
X = cell(num_v,1);
for i = 1:num_v
    [X{i},Y] = twomoon_gen(num_x/2, num_x/2, (i+4)/35, -0.8, 0.05);
end
% % add an noise view
% num_v = num_v + 1;
% mu=[1,1];
% sigma = [0.1,0;0,0.1];
% X{num_v} = mvnrnd(mu,sigma,num_x);

Y_label = zeros(num_x,num_c); % ground truth
for i = 1:num_x
    Y_label(i,Y(i)+1) = 1;
end

% visualization
figure(1)
for i = 1:num_v
    subplot(2,2,i);
    scatter(X{i}(:,1),X{i}(:,2),200,[zeros(num_x,1),Y_label],'.')
end

% % generate toy datasets
% num_x = 300;
% num_c = 3;
% num_v = 3;
% Y_label = zeros(num_x,num_c);
% Y_label(1:100,1) = 1;
% Y_label(101:200,2) = 1;
% Y_label(201:300,3) = 1;
% X = cell(num_v,1);
% x = zeros(num_x,2);
% for i = 1:num_v
% mu=[1,1];
% sigma = [0.1,0;0,0.1];
% x([(100*(i-1)+1):100*i],:)=mvnrnd(mu,sigma,100);
% mu1=[-1,-0.8];
% mu2=[-1,-1.2];
% x([1:100*(i-1),100*i+1:num_x],:)=[mvnrnd(mu1,sigma,100);mvnrnd(mu2,sigma,100)];
% X{i} = x;
% end
% 
% % % add a noise view
% % num_v = num_v+1;
% % mu=[1,1];
% % sigma = [0.1,0;0,0.1];
% % X{num_v} = mvnrnd(mu,sigma,num_x);
% 
% [~,Y] = max(Y_label,[],2); % ground truth
% % visualization
% figure(1)
% for i = 1:num_v
%     subplot(2,2,i);
%     scatter(X{i}(:,1),X{i}(:,2),200,[Y_label],'.')
% end


%% construct anchor graphs
num_m = 100;  % number of anchors
num_k = 5;  % number of k nearest neighbours

B = cell(num_v,2);
for i = 1:num_v
    [B{i,1}] = GraphConstruction(X{i},num_m,num_k);
end




%% 聚类

% initialization
Z0 = cell(num_v,1);
for i = 1:num_v
    num_m = size(B{i,1},2);
    Z0{i} = randn(num_m,num_c);
    Z0{i} = Z0{i} - min(Z0{i},[],2);
    Z0{i} = projectm(Z0{i});
end
Y0 = randn(num_x,num_c);
Y0 = Y0 - min(Y0,[],2);
Y0 = projectm(Y0);


%%

diff0 = -1;
pY0 = Y0;
pZ0 = Z0;
obj0 = zeros(201);

lambda = 2.7; % regularization parameter

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
    [pY,pZ,obj,diff,regu,di] = MVFCAG(B,Y0,Z0,lambda,num_c);
    objval = diff + regu;
    if objval<objval0
        objval0 = objval;
        pY0 = pY;
        pZ0 = pZ;
        obj0 = obj;
        di0 = di;
    end
%     if objval0 <-51.9
%         break;
%     end
end


% visualization
[~,predY] = max(pY0,[],2);
result = ClusteringMeasure(Y, predY);
if(num_c == 2)
    pY0 = [zeros(num_x,1),pY0];
end
figure(2)
for i = 1:num_v
    subplot(2,2,i);
    scatter(X{i}(:,1),X{i}(:,2),200,[pY0],'.')
end

figure(3)
for i = 1:num_v
    subplot(1,num_v,i);
    imshow(full(B{i}).*255);
    title(sprintf('d%d = %f',i,di0(i)),'FontSize',14);
end

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
