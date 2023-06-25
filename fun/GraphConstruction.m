function [B] = GraphConstruction(x,num_m,num_k)
%GRAPHCONSTRUCTION �˴���ʾ�йش˺�����ժҪ
%x:����
%num_m:ê����Ŀ
%num_k:k����

num_x = size(x,1);
% k��ֵѡȡê��
[~,U] = kmeans(x,num_m);
% �������ͼ�ڽӾ���
B = zeros(num_x,num_m); %����ͼ�ڽӾ����ʼ��
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


