function [B] = GraphConstruction(x,num_m,num_k)
%GRAPHCONSTRUCTION 此处显示有关此函数的摘要
%x:样本
%num_m:锚点数目
%num_k:k近邻

num_x = size(x,1);
% k均值选取锚点
[~,U] = kmeans(x,num_m);
% 计算二部图邻接矩阵
B = zeros(num_x,num_m); %二部图邻接矩阵初始化
for i=1:num_x
    d = zeros(1,num_m);
    for j = 1:num_m
        d(1,j) = (x(i,:)-U(j,:))*(x(i,:)-U(j,:))';
    end  
    [e,didx] = sort(d);
    di = e(1,1:num_k+1);
    id = didx(1,1:num_k+1);
    B(i,id) = (di(num_k+1)-di)/(num_k*di(num_k+1)-sum(di(1:num_k))+eps);
end

[~,U_ind] = max(B,[],1);
[U_ind,B_ind] = sort(U_ind);
B = B(:,B_ind);
B = sparse(B);

end


