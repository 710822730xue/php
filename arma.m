s = 24; %周期是24
x = t_test';%初始数据的录入
n = 168; %预报的个数
m1 = length(x); %原始的数据的个数
for i = s+1:m1
    y(i-s) = x(i) - x(i-s);%进行周期差分变换
end
w = diff(y); %消除趋势性的差分运算
m2 = length(2);
for i = 0:6
    for j = 0:6
        if i == 0 && j == 0
            continue
        elseif i == 0
            ToEstMd = arima('MALags',1:j,'Constant',0); %指定模型的结构
        elseif j == 0
            ToEstMd = arima('ARLags',1:i,'Constant',0); %指定模型的结构
        else
            ToEstMd = arima('ARLags',1:i,'MALags',1:j,'Constant',0); %指定模型的结构
        end
        k=0;
        k = k + 1;
        R(k) = i;
        M(k) = j;
        [EstMd,EstParamCov,LogL,info] = estimate(ToEstMd,w');%模型拟合
        numParams = sum(any(EstParamCov));%计算拟合参数的个数
        [aic(k),bic(k)] = aicbic(LogL,numParams,m2);
    end
end
fprintf('R,M,AIC,BIC的对应值如下\n%f');%显示计算结果
check  = [R',M',aic',bic'];
%%
x = t_test'

ToEstMd = arima('ARLags',1:5,'MALags',1:2,'Constant',0);%指定模型的结构
[EstMd,EstParamCov,LogL,info] = estimate(ToEstMd,w');%模型拟合 
w_Forecast = forecast(EstMd,n,'Y0',w');
yhat = y(end) + cumsum(w_Forecast); %一阶差分的还原值
for j = 1:n
    x(m1 + j) = yhat(j) + x(m1+j-s); %x的预测值
end
x(m1+1:end);
