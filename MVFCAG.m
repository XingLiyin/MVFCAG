%input 二部图矩阵B 参数lamda
function [Y,Z,obj,diff,regu,di] = MVFCAG(B,Y0, Z0,lambda,num_c)

num_x = size(B{1,1},1); % 样本数目
num_v = size(B,1); %视图数目
h = 1e-2;  % 梯度下降的步长
th = 1e-3; % 梯度下降结束条件
maxiter = 200;  
MAXITER = 200;
% 初始化
Y = Y0;
Z = Z0;

di = zeros(num_v,1); % reweighted系数
obj = zeros(MAXITER+1,1); % 目标函数值
for i = 1:num_v
    obj(1) = obj(1) + norm(B{i,1}*Z{i}-Y,'fro');
end
obj(1) = obj(1) - lambda*trace(sqrtm(Y'*Y));

% I = eye(num_c);
% O = zeros((num_x-num_c),num_c);
% E = [I;O];


for iter = 1:MAXITER
    
    % update Y (ReWeighted)
    Yt1 = Y;  % 初始化
    for t_rw = 1:maxiter
        Yt0 = Yt1; % save
        % calculate d
        for i = 1:num_v
            di(i) = 1.0/(2*norm(B{i,1}*Z{i}-Yt0,'fro'));
        end
%         [U,~, V] = svd(Y);
%         D = U*E*V';
        D = Yt0/(sqrtm(Yt0'*Yt0));
        % calculate Y
        Yt1 = zeros(num_x,num_c);
        for i = 1:num_v
            Yt1 = Yt1 + di(i)*B{i,1}*Z{i};
        end
        Yt1 = Yt1 + 0.5*lambda*D;
        Yt1 = Yt1/sum(di);
        Yt1 = projectm(Yt1);
        % 提前截止
        if(norm(Yt1-Yt0,'fro') < th) 
            break;
        end   
    end
    Y = Yt1; % 更新
    
    % update Z
    parfor j = 1:num_v
        
%         Z{j} = mexUpdateZi(Y,B{j,1},Z{j},j);
        
        % 坐标下降法
        num_m = size(B{j,1},2);
        Zi1 = Z{j}'; % initialize 转置为进行更高效的运算
        for t_cd = 1:1
            Zi0 = Zi1; % save
            for i = 1:num_m
                Br = B{j,1};
                Zi1(:,i) = 0;
                bi = B{j,1}(:,i);
                zi = ((Y - Br*Zi1')'*bi)/(bi'*bi);
                zi = projectv(zi);
                Zi1(:,i) = zi;
            end
        end
        Z{j} = Zi1'; % update
    end
    
    % 目标函数值
    diff = 0;
    for i = 1:num_v
        diff = diff + norm(B{i,1}*Z{i}-Y,'fro');
    end
    regu = -lambda*trace(sqrtm(Y'*Y));
    obj(iter+1) = diff + regu;
    % 提前截止
    if(abs(obj(iter+1) - obj(iter))<th)
        fprintf('The Value of Objective Function:%f\n',obj(iter+1))
        break;
    end
    
    if(iter>30 && abs(obj(iter+1) / obj(iter)-1)<th)
        fprintf('The Value of Objective Function:%f\n',obj(iter+1))
        break;
    end

end

end

function x = projectv(v)
alpha0 = max(v); 
alpha1 = min(alpha0-1,min(v));
while abs(alpha1-alpha0) > 1e-4
    alpha0 = alpha1; % save
    f = sum(v(v>alpha0) - alpha0)-1; 
    df = -sum((v>alpha0));
    alpha1 = alpha0 - f/df; % update
end
x = v - alpha1;
x(x<0) = 0;
end

function x = projectm(m)
alpha0 = max(m,[],2); 
alpha1 = min(m,[],2); %初始化
while any(abs(alpha1-alpha0) > 1e-4)
    alpha0 = alpha1; % save
    f = sum((m - alpha0).*(m>alpha0),2)-1; 
    df = -sum((m>alpha0),2);
    alpha1 = alpha0 - f./df; % update
end
x = m - alpha1;
x(x<0) = 0;
end