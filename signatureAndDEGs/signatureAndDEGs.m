clc;clear;close all; 
file_path = 'D:\SignatureAndDEGs\';


file = dir([file_path,'total_miRNA_gene.xlsx']);

% info_lib = readcell(strcat(file(1).folder,'\','p-value.xlsx'));
info_lib = readcell(strcat(file(1).folder,'\','EL-CAML.xlsx'));
miRNA_tmp = replace(string(info_lib),'mir','miR');;

[a,~] = size(file);
data_record = [];

info = readcell(strcat(file(1).folder,'\',file(1).name));
strname = replace(file(1).name,'.xlsx','');

input_raw = info(2:end,:);
input_TF = find(str2double(string(input_raw(:,10)))<= 17);
input_ = input_raw(input_TF,:);

input = input_;
        
% find all miRNA 
[~,idx] = unique(string(input(:,2)),'rows','first');
miRNA_id = input(sort(idx),2);
    

% select n miRNA and count performance ==========================================
select = 28;
for iii = 1: 30
    a=randperm(size(miRNA_id,1));
    miRNA_tmp = miRNA_id(a(1:select));
    
    % select miRNA target to gene
    record_tmp = [];
    for ii = 1: size(miRNA_tmp,1)
        TF_tmp = find(string(input(:,2)) == string(miRNA_tmp(ii) ));
        record_tmp = [record_tmp; input(TF_tmp,:)];
    end

    count_gene = [ record_tmp(:,4) record_tmp(:,17)];
    [~,idx_tmp] = unique(string(count_gene(:,1)),'rows','first');
    miRNA_select_tmp = count_gene(sort(idx_tmp),:);
    
    % uni gene count gene#
    TF_tmp_15 = find(abs(str2double(string(miRNA_select_tmp(:,2))))>= 1.3);
    TF_tmp_20 = find(abs(str2double(string(miRNA_select_tmp(:,2))))>= 1.5);
    TF_tmp_30 = find(abs(str2double(string(miRNA_select_tmp(:,2))))>= 2.0);
    % select miRNA name
    miRNA_name = replace(string(miRNA_tmp),'hsa-miR-','');
    miRNA_name_op = [];
    for j = 1:size(miRNA_name,1)
        miRNA_name_op = strcat(miRNA_name_op,miRNA_name(j),',');
    end
    
    % record miRNA name & perform
    data_record = [ data_record; num2cell((iii)) cellstr(miRNA_name_op) num2cell(size(miRNA_select_tmp,1)) ...
        num2cell(size(TF_tmp_15,1)) num2cell(size(TF_tmp_20,1)) num2cell(size(TF_tmp_30,1))];
end


writecell(data_record,strcat(file_path,strname,'output.xlsx'));


% output = readcell(strcat(file_path,'output.xlsx'));
perform =data_record(:,end-3:end);

ibcgavalue = perform(1,:);
pvalue = perform(2,:);
noisevalue = perform(3:end,:);

record = [];
for jj = 1:4
    [h2,p2,ci2,stats2] = ttest(noisevalue(:,jj),ibcgavalue(:,jj))
    [h3,p3,ci3,stats3] = ttest(noisevalue(:,jj),pvalue(:,jj))
    record = [record; h2 p2 ci2' h3 p3 ci3' ];
end

writematrix(record,strcat(file_path,strname,'significant_output.xlsx'));
