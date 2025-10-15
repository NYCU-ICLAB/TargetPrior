if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(c("limma", "Biobase"))


library(limma) ## function library for modified t-statistic
library(Biobase) ## function library to modify the imported dataset
setwd("D:/logFC/mRNA/")

raw <- read.table (file="TARGET-AML_mRNA.csv",sep=',',  row.names=1,header=T)
input_data <- as.matrix(raw)
input <- input_data

## Only for importing data with log2 values
eset<-new('ExpressionSet',exprs=as.matrix(input))
eset<-new('ExpressionSet',exprs=input)
design <- model.matrix (~-1+factor(c(rep(1,241),rep(2,25))))
colnames(design) <- c('re','normal') ##Group1 = normal , Group2 = tumor
fit <- lmFit(eset,design=design) ##Fit the format of limma function
contrast.matrix <-makeContrasts(re-normal, levels=design)
fit2<-contrasts.fit(fit,contrast.matrix)
P <- eBayes(fit2) 
write.table(topTable(P,n=1000000,adjust='BH'),file="mRNA_BHmethod.csv",sep=',',quote=F)
##(Benjamini and HochbergA!A|s method)

# logFC: log2 fold change of I5.9/I5.6
# AveExpr: Average expression across all samples, in log2 CPM
# t: logFC divided by its standard error
# P.Value: Raw p-value (based on t) from test that logFC differs from 0
# adj.P.Val: Benjamini-Hochberg false discovery rate adjusted p-value
# B: log-odds that gene is DE (arguably less useful than the other columns)


