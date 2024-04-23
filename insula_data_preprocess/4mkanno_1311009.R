ASC_resolution=0.09
EX_resolution=1.3
IN_resolution=1.1
NON_resolution=0.09
MICRO_resolution=0.07
OPC_resolution=0.07


suppressMessages(library(tidyverse))
suppressMessages(library(parallel))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))
suppressMessages(library(patchwork))
suppressMessages(library(Matrix))
suppressMessages(library(viridis))
suppressMessages(library(cowplot))
suppressMessages(library(ggsci))
suppressMessages(library(pheatmap))
suppressMessages(library(RColorBrewer))
suppressMessages(library(reshape2))
suppressMessages(library(LSD))
suppressMessages(library(readxl))
suppressMessages(library(ggrepel))
suppressMessages(library(randomForest))
suppressMessages(library(harmony))

options(future.globals.maxSize= 1000 * 1024^16 )

dataDir=paste0('/home/liuyx/liuyuxuan/spa_seurat/cla/speciesCompare/Insular_renew_1016/batch_normalize_renew_240115/dif_res_anno/')

classFiltSeurat=readRDS('/home/liuyx/liuyuxuan/spa_seurat/cla/speciesCompare/Insular_renew_1016/batch_normalize_renew_240115/insular_snrna_all_soupX_fil300_800_anno_SCT_RFfil_240116.rds')
classFiltSeurat

classFiltSeurat
table(classFiltSeurat@meta.data$maxPredClass)

Excit=subset(classFiltSeurat,maxPredClass=='Excit_Neuron')
Inhibit=subset(classFiltSeurat,maxPredClass=='Inhibit_Neuron')
Astrocytes=subset(classFiltSeurat,maxPredClass=='Astrocytes')
Oligo=subset(classFiltSeurat,maxPredClass=='Oligo')
Microglia=subset(classFiltSeurat,maxPredClass=='Microglia')
OPC=subset(classFiltSeurat,maxPredClass=='OPC')
#MSN=subset(classFiltSeurat,maxPredClass=='MSN')
#VLMC=subset(classFiltSeurat,maxPredClass=='VLMC')
Endotheial=subset(classFiltSeurat,maxPredClass=='Endotheial')



#剔除线粒体基因
assay=Excit@assays$RNA@counts
meta=Excit@meta.data
mt_gene=c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
new_assay=assay[setdiff(rownames(assay),mt_gene),]

Excit1=CreateSeuratObject(counts  = new_assay)
Excit1=AddMetaData(Excit1,meta)

Excit1 =  NormalizeData(Excit1, normalization.method = "LogNormalize", scale.factor = 10000)
Excit1 = FindVariableFeatures(Excit1, selection.method = "vst", nfeatures = 3000)
Excit1 = ScaleData(Excit1, features = rownames(Excit1))
Excit1 = RunHarmony(
    Excit1, group.by.vars="sample", plot_convergence=T,
    reduction = "pca", dims.use=1:30,
    theta=2,)
Excit1 = Excit1 %>%
    FindNeighbors(reduction="harmony", dims = 1:30) %>%
    FindClusters(verbose = FALSE,resolution = EX_resolution) %>%
    RunUMAP(reduction="harmony", dims = 1:30)


#剔除线粒体基因
assay=Inhibit@assays$RNA@counts
meta=Inhibit@meta.data
mt_gene=c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
new_assay=assay[setdiff(rownames(assay),mt_gene),]

Inhibit1=CreateSeuratObject(counts  = new_assay)
Inhibit1=AddMetaData(Inhibit1,meta)

Inhibit1 =  NormalizeData(Inhibit1, normalization.method = "LogNormalize", scale.factor = 10000)
Inhibit1 = FindVariableFeatures(Inhibit1, selection.method = "vst", nfeatures = 3000)
Inhibit1 = ScaleData(Inhibit1, features = rownames(Inhibit1))
Inhibit1 = RunHarmony(
    Inhibit1, group.by.vars="sample", plot_convergence=T,
    reduction = "pca", dims.use=1:30,
    theta=2,)
Inhibit1 = Inhibit1 %>%
    FindNeighbors(reduction="harmony", dims = 1:30) %>%
    FindClusters(verbose = FALSE,resolution = IN_resolution) %>%
    RunUMAP(reduction="harmony", dims = 1:30)


#剔除线粒体基因
assay=Astrocytes@assays$RNA@counts
meta=Astrocytes@meta.data
mt_gene=c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
new_assay=assay[setdiff(rownames(assay),mt_gene),]

Astrocytes1=CreateSeuratObject(counts  = new_assay)
Astrocytes1=AddMetaData(Astrocytes1,meta)

Astrocytes1 =  NormalizeData(Astrocytes1, normalization.method = "LogNormalize", scale.factor = 10000)
Astrocytes1 = FindVariableFeatures(Astrocytes1, selection.method = "vst", nfeatures = 3000)
Astrocytes1 = ScaleData(Astrocytes1, features = rownames(Astrocytes1))
Astrocytes1 = RunHarmony(
    Astrocytes1, group.by.vars="sample", plot_convergence=T,
    reduction = "pca", dims.use=1:30,
    theta=2,)
Astrocytes1 = Astrocytes1 %>%
    FindNeighbors(reduction="harmony", dims = 1:30) %>%
    FindClusters(verbose = FALSE,resolution = ASC_resolution) %>%
    RunUMAP(reduction="harmony", dims = 1:30)


#剔除线粒体基因
assay=Oligo@assays$RNA@counts
meta=Oligo@meta.data
mt_gene=c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
new_assay=assay[setdiff(rownames(assay),mt_gene),]

Oligo1=CreateSeuratObject(counts  = new_assay)
Oligo1=AddMetaData(Oligo1,meta)

Oligo1 =  NormalizeData(Oligo1, normalization.method = "LogNormalize", scale.factor = 10000)
Oligo1 = FindVariableFeatures(Oligo1, selection.method = "vst", nfeatures = 3000)
Oligo1 = ScaleData(Oligo1, features = rownames(Oligo1))
Oligo1 = RunHarmony(
    Oligo1, group.by.vars="sample", plot_convergence=T,
    reduction = "pca", dims.use=1:30,
    theta=2,)
Oligo1 = Oligo1 %>%
    FindNeighbors(reduction="harmony", dims = 1:30) %>%
    FindClusters(verbose = FALSE,resolution = NON_resolution) %>%
    RunUMAP(reduction="harmony", dims = 1:30)


#剔除线粒体基因
assay=OPC@assays$RNA@counts
meta=OPC@meta.data
mt_gene=c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
new_assay=assay[setdiff(rownames(assay),mt_gene),]

OPC1=CreateSeuratObject(counts  = new_assay)
OPC1=AddMetaData(OPC1,meta)

OPC1 =  NormalizeData(OPC1, normalization.method = "LogNormalize", scale.factor = 10000)
OPC1 = FindVariableFeatures(OPC1, selection.method = "vst", nfeatures = 3000)
OPC1 = ScaleData(OPC1, features = rownames(OPC1))
OPC1 = RunHarmony(
    OPC1, group.by.vars="sample", plot_convergence=T,
    reduction = "pca", dims.use=1:30,
    theta=2,)
OPC1 = OPC1 %>%
    FindNeighbors(reduction="harmony", dims = 1:30) %>%
    FindClusters(verbose = FALSE,resolution = OPC_resolution) %>%
    RunUMAP(reduction="harmony", dims = 1:30)



#剔除线粒体基因
assay=Endotheial@assays$RNA@counts
meta=Endotheial@meta.data
mt_gene=c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
new_assay=assay[setdiff(rownames(assay),mt_gene),]

Endotheial1=CreateSeuratObject(counts  = new_assay)
Endotheial1=AddMetaData(Endotheial1,meta)

Endotheial1 =  NormalizeData(Endotheial1, normalization.method = "LogNormalize", scale.factor = 10000)
Endotheial1 = FindVariableFeatures(Endotheial1, selection.method = "vst", nfeatures = 3000)
Endotheial1 = ScaleData(Endotheial1, features = rownames(Endotheial1))
Endotheial1 = RunHarmony(
    Endotheial1, group.by.vars="sample", plot_convergence=T,
    reduction = "pca", dims.use=1:30,
    theta=2,)
Endotheial1 = Endotheial1 %>%
    FindNeighbors(reduction="harmony", dims = 1:30) %>%
    FindClusters(verbose = FALSE,resolution = NON_resolution) %>%
    RunUMAP(reduction="harmony", dims = 1:30)

#剔除线粒体基因
assay=Microglia@assays$RNA@counts
meta=Microglia@meta.data
mt_gene=c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
new_assay=assay[setdiff(rownames(assay),mt_gene),]

Microglia1=CreateSeuratObject(counts  = new_assay)
Microglia1=AddMetaData(Microglia1,meta)

Microglia1 =  NormalizeData(Microglia1, normalization.method = "LogNormalize", scale.factor = 10000)
Microglia1 = FindVariableFeatures(Microglia1, selection.method = "vst", nfeatures = 3000)
Microglia1 = ScaleData(Microglia1, features = rownames(Microglia1))
Microglia1 = RunHarmony(
    Microglia1, group.by.vars="sample", plot_convergence=T,
    reduction = "pca", dims.use=1:30,
    theta=2,)
Microglia1 = Microglia1 %>%
    FindNeighbors(reduction="harmony", dims = 1:30) %>%
    FindClusters(verbose = FALSE,resolution = MICRO_resolution) %>%
    RunUMAP(reduction="harmony", dims = 1:30)

Astrocytes1@meta.data$Subclass=paste0(Astrocytes1@meta.data$maxPredClass,'_',Astrocytes1@meta.data$seurat_clusters)
Oligo1@meta.data$Subclass=paste0(Oligo1@meta.data$maxPredClass,'_',Oligo1@meta.data$seurat_clusters)
Microglia1@meta.data$Subclass=paste0(Microglia1@meta.data$maxPredClass,'_',Microglia1@meta.data$seurat_clusters)
OPC1@meta.data$Subclass=paste0(OPC1@meta.data$maxPredClass,'_',OPC1@meta.data$seurat_clusters)
#MSN1@meta.data$Subclass=paste0(MSN1@meta.data$maxPredClass,'_',MSN1@meta.data$seurat_clusters)
Endotheial1@meta.data$Subclass=paste0(Endotheial1@meta.data$maxPredClass,'_0')
Excit1@meta.data$Subclass=paste0(Excit1@meta.data$maxPredClass,'_',Excit1@meta.data$seurat_clusters)
Inhibit1@meta.data$Subclass=paste0(Inhibit1@meta.data$maxPredClass,'_',Inhibit1@meta.data$seurat_clusters)




sub1=Astrocytes1@meta.data[,'Subclass',drop=F]
sub2=Oligo1@meta.data[,'Subclass',drop=F]
sub3=Microglia1@meta.data[,'Subclass',drop=F]
sub4=OPC1@meta.data[,'Subclass',drop=F]
#sub5=MSN1@meta.data[,'Subclass',drop=F]
sub6=Excit1@meta.data[,'Subclass',drop=F]
sub7=Inhibit1@meta.data[,'Subclass',drop=F]
#sub8=VLMC@meta.data[,'Subclass',drop=F]
sub9=Endotheial1@meta.data[,'Subclass',drop=F]

sub_2=rbind(sub1,sub2,sub3,sub4,sub6,sub7,sub9)

df_dir=dataDir

classFiltSeurat=AddMetaData(classFiltSeurat,sub_2,col.name = 'Subclass')

classFiltSeurat
table(classFiltSeurat$Subclass)

df_dir=dataDir
save_file_name=paste0(df_dir,EX_resolution,'_res',IN_resolution,'_res',NON_resolution,'_anno.csv')
write.csv(sub_2,save_file_name)


classFiltSeurat=AddMetaData(classFiltSeurat,sub_2,col.name = 'Subclass')

classFiltSeurat
table(classFiltSeurat$Subclass)

# 每个class抽样80%，最多3000个细胞
seurat=classFiltSeurat
sampleRatio = 0.8
maxCell = 3000
sampleDf = seurat@meta.data[, c("Subclass"), drop=F]
sampleList = split(sampleDf, sampleDf$Subclass)
sampleList = lapply(sampleList, function(x) {
    sampleSize = as.integer(nrow(x)*0.8)
    if (sampleSize > maxCell) {
        sampleSize = maxCell
    }
    sampleIdx = 1:nrow(x)
    sampleIdx = sample(1:nrow(x), size=sampleSize, replace=F)
    return(x[sampleIdx,,drop=F])
})
sampleDf = Reduce(rbind, sampleList)
str(sampleDf)
table(sampleDf$class)

trainSeurat = seurat[,rownames(sampleDf)]
table(trainSeurat$Subclass)

saveDir=paste0(dataDir,'train_seurat/')
#saveRDS(trainSeurat,paste0(saveDir,'res',EX_resolution,'_res',IN_resolution,'_traindata.rds'))
#trainSeurat=readRDS(paste0(saveDir,'res',EX_resolution,'_res',IN_resolution,'_traindata.rds'))
Idents(trainSeurat) = "Subclass"
plan("multicore", workers = 5)
classMarker = FindAllMarkers(trainSeurat)
write_tsv(classMarker, paste0(saveDir, "/train_res",EX_resolution,'_res',IN_resolution,'_res',NON_resolution,"_downsample3000_subclass_marker.tsv"))


classMarker=read.table(paste0(saveDir, "/train_res",EX_resolution,'_res',IN_resolution,'_res',NON_resolution,"_downsample3000_subclass_marker.tsv"))
colnames(classMarker)=classMarker[1,]
classMarker=classMarker[-1,]

#classMarker = read_tsv(paste0(dataDir, "/allSnRNAseq.soupx.mergeSeuratFiltlibFilt800mt5Ratio1d2.downsample5000ClassMarker.20220822.tsv"))
classMarker = subset(classMarker, p_val_adj < 0.01 & avg_log2FC > 1)
table(classMarker$cluster)


classMarkerTop = classMarker %>% group_by(cluster) %>% top_n(n = 20, wt = avg_log2FC)
options(repr.plot.width=40, repr.plot.height=10)
#Idents(seurat) = "Subclass"
#DotPlot(seurat, features=unique(c(classMarkerTop$gene))) + RotatedAxis()

trainDf = as.data.frame(t(as.matrix(trainSeurat[["SCT"]]@data[unique(classMarker$gene), ])))
trainDf$Subclass = trainSeurat$Subclass
trainDf$Subclass = factor(trainDf$Subclass)



# We need to modify the name of the columns
colnames(trainDf) = paste0("col_", colnames(trainDf))
colnames(trainDf) = gsub("-", "_", colnames(trainDf))

rf = randomForest(col_Subclass~., data=trainDf, ntree=500)

saveRDS(rf, paste0(saveDir, "/Subclass.randomForest.20230627res",EX_resolution,'_res',IN_resolution,".rds"))


