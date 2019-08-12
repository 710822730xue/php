s = 24; %������24
x = t_test';%��ʼ���ݵ�¼��
n = 168; %Ԥ���ĸ���
m1 = length(x); %ԭʼ�����ݵĸ���
for i = s+1:m1
    y(i-s) = x(i) - x(i-s);%�������ڲ�ֱ任
end
w = diff(y); %���������ԵĲ������
m2 = length(2);
for i = 0:6
    for j = 0:6
        if i == 0 && j == 0
            continue
        elseif i == 0
            ToEstMd = arima('MALags',1:j,'Constant',0); %ָ��ģ�͵Ľṹ
        elseif j == 0
            ToEstMd = arima('ARLags',1:i,'Constant',0); %ָ��ģ�͵Ľṹ
        else
            ToEstMd = arima('ARLags',1:i,'MALags',1:j,'Constant',0); %ָ��ģ�͵Ľṹ
        end
        k=0;
        k = k + 1;
        R(k) = i;
        M(k) = j;
        [EstMd,EstParamCov,LogL,info] = estimate(ToEstMd,w');%ģ�����
        numParams = sum(any(EstParamCov));%������ϲ����ĸ���
        [aic(k),bic(k)] = aicbic(LogL,numParams,m2);
    end
end
fprintf('R,M,AIC,BIC�Ķ�Ӧֵ����\n%f');%��ʾ������
check  = [R',M',aic',bic'];
%%
x = t_test'

ToEstMd = arima('ARLags',1:5,'MALags',1:2,'Constant',0);%ָ��ģ�͵Ľṹ
[EstMd,EstParamCov,LogL,info] = estimate(ToEstMd,w');%ģ����� 
w_Forecast = forecast(EstMd,n,'Y0',w');
yhat = y(end) + cumsum(w_Forecast); %һ�ײ�ֵĻ�ԭֵ
for j = 1:n
    x(m1 + j) = yhat(j) + x(m1+j-s); %x��Ԥ��ֵ
end
x(m1+1:end);
