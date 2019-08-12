% ѵ����
[pn_train,inputps] = mapminmax(p_train');
pn_train = pn_train';
pn_test = mapminmax('apply',p_test',inputps);
pn_test = pn_test';
% ���Լ�
[tn_train,outputps] = mapminmax(t_train');
tn_train = tn_train';
tn_test = mapminmax('apply',t_test',outputps);
tn_test = tn_test';
%% ѡ��GA��ѵ�SVM����c&g
% GA�Ĳ���ѡ���ʼ��
ga_option.maxgen = 100;
ga_option.sizepop = 50; 
ga_option.cbound = [0,100];
ga_option.gbound = [0,100];
ga_option.v = 5;
ga_option.ggap = 0.9;
[BestCCaccuracy,Bestc,Bestg,ga_option] = gaSVMcgForClass(pn_train,tn_train,ga_option);
%% ������ѵĲ�������SVM����ѵ��
% ����/ѵ��SVM  
cmd = [' -t 2',' -c ',num2str(Bestc),' -g ',num2str(Bestg),' -s 3 -p 0.01'];
model = svmtrain(tn_train,pn_train,cmd);
%% SVM����Ԥ��
[Predict_1,error_1,decision_values] = svmpredict(tn_train,pn_train,model);
[Predict_2,error_2,decision_values] = svmpredict(tn_test,pn_test,model);
% ����һ��
predict_1 = mapminmax('reverse',Predict_1,outputps);
predict_2 = mapminmax('reverse',Predict_2,outputps);
% ����Ա�
result_1 = [t_train predict_1];
result_2 = [t_test predict_2];
%% ��ͼ
figure(1)
plot(1:length(t_train),t_train,'r-*',1:length(t_train),predict_1,'b:o')
grid on
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Сʱ��')
ylabel('���')
string_1 = {'ѵ����Ԥ�����Ա�';
           ['mse = ' num2str(error_1(2)) ' R^2 = ' num2str(error_1(3))]};
title(string_1)
figure(2)
plot(1:length(t_test),t_test,'r-*',1:length(t_test),predict_2,'b:o')
grid on
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Сʱ��')
ylabel('���')
string_2 = {'���Լ�Ԥ�����Ա�';
           ['mse = ' num2str(error_2(2)) ' R^2 = ' num2str(error_2(3))]};
title(string_2)
%% �����Ŵ��㷨�����Ż���һ���Ӻ���
function [BestCCaccuracy,Bestc,Bestg,ga_option] = gaSVMcgForClass(pn_train,tn_train,ga_option)
% gaSVMcgForClass
% ������ʼ��
if nargin == 2
    ga_option = struct('maxgen',100,'sizepop',50,'ggap',0.9,...
        'cbound',[0,100],'gbound',[0,100],'v',5);
end
% maxgen:���Ľ�������,Ĭ��Ϊ200,һ��ȡֵ��ΧΪ[100,500]
% sizepop:��Ⱥ�������,Ĭ��Ϊ20,һ��ȡֵ��ΧΪ[20,100]
% cbound = [cmin,cmax],����c�ı仯��Χ,Ĭ��Ϊ(0,100]
% gbound = [gmin,gmax],����g�ı仯��Χ,Ĭ��Ϊ[0,1000]
% v:SVM Cross Validation����,Ĭ��Ϊ5

%
MAXGEN = ga_option.maxgen;
NIND = ga_option.sizepop;
NVAR = 2;
PRECI = 20;
GGAP = ga_option.ggap;
trace = zeros(MAXGEN,2);

FieldID = ...
[rep((PRECI),[1,NVAR]);[ga_option.cbound(1),ga_option.gbound(1);ga_option.cbound(2),ga_option.gbound(2)]; ...
 [1,1;0,0;0,1;1,1]];

Chrom = crtbp(NIND,NVAR*PRECI);

gen = 1;
v = ga_option.v;
BestCCaccuracy = 0;
Bestc = 0;
Bestg = 0;
%
cg = bs2rv(Chrom,FieldID);

for nind = 1:NIND
    cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2))];
    ObjV(nind,1) = svmtrain(pn_train,tn_train,cmd);
end
[BestCCaccuracy,I] = max(ObjV);
Bestc = cg(I,1);
Bestg = cg(I,2);

for gen = 1:MAXGEN
    FitnV = ranking(-ObjV);
    
    SelCh = select('sus',Chrom,FitnV,GGAP);
    SelCh = recombin('xovsp',SelCh,0.7);
    SelCh = mut(SelCh);
    
    cg = bs2rv(SelCh,FieldID);
    for nind = 1:size(SelCh,1)
        cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2))];
        ObjVSel(nind,1) = svmtrain(pn_train,tn_train,cmd);
    end
    
    [Chrom,ObjV] = reins(Chrom,SelCh,1,1,ObjV,ObjVSel);
    
    if max(ObjV) <= 50
        continue;
    end
    
    [NewBestCVaccuracy,I] = max(ObjV);
    cg_temp = bs2rv(Chrom,FieldID);
    temp_NewBestCVaccuracy = NewBestCVaccuracy;
    
    if NewBestCVaccuracy > BestCCaccuracy
       BestCCaccuracy = NewBestCVaccuracy;
       Bestc = cg_temp(I,1);
       Bestg = cg_temp(I,2);
    end
    
    if abs( NewBestCVaccuracy-BestCCaccuracy ) <= 10^(-2) && ...
        cg_temp(I,1) < Bestc
       BestCCaccuracy = NewBestCVaccuracy;
       Bestc = cg_temp(I,1);
       Bestg = cg_temp(I,2);
    end    
    
    trace(gen,1) = max(ObjV);
    trace(gen,2) = sum(ObjV)/length(ObjV);
    gen=gen+1;
    if gen<=MAXGEN/2
        continue;
    end
   
    if BestCCaccuracy >=80 && ... 
       ( temp_NewBestCVaccuracy-BestCCaccuracy ) <= 10^(-2)      
        break; 
    end 
    if gen == MAXGEN 
        break; 
    end 
     
end 
gen= gen -1; 
  
end

