clc;clear;close all; 
file_path = 'D:\SignatureAndDEGs\';

input = readcell(strcat(file_path,'output.xlsx'));
perform = input(:,end-3:end);
ibcgavalue = perform(1,:);
pvalue = perform(2,:);
noisevalue = perform(3:end,:);
title = ['DEG |log_2FC|>1 (count)','DEG |log_2FC|>1.3 (count)','DEG |log_2FC|>1.5 (count)', 'DEG |log_2FC|>2.0 (count)'];

rows = 1;cols = 3;
for i = 2:4
    subplot(rows,cols,i-1)
    
    plot(ones(length(ibcgavalue),1),ibcgavalue(:,i),'Marker','o','LineWidth',2,'MarkerEdgeColor','#6096BA');
    hold on
    plot(2*ones(length(pvalue),1),pvalue(:,i),'Marker','o','LineWidth',2,'MarkerEdgeColor','#0b2545');
    hold on
    plot(3*ones(length(noisevalue),1),noisevalue(:,i),'Marker','o','LineWidth',2,'MarkerEdgeColor','#00b4d8');
    hold on
    
    max_value = max(perform(:,2:4));
    axis([0.5 3.5 0 max_value(1)+50])
    yticks(0:50:max_value(1)+50)
    xticklabels({'EL-CAML', 'p-value','Background'})
    set(gca,'FontSize',14);
    ylabel(string(title(i)),'fontsize',16);
    
end

set(gcf,'Position',[2367,186,1087,668])

img = getimage(gcf); 
print(img, '-dtiffn', '-r600', [file_path,'scatter_plot.tiff']) 

saveas(gcf,[file_path,'scatter_plot.jpg'],'jpg')
close all; 