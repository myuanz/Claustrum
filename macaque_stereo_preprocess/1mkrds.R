suppressMessages(library(tidyverse))
suppressMessages(library(parallel))
suppressMessages(library(Seurat))
#suppressMessages(library(SeuratData))
suppressMessages(library(patchwork))
suppressMessages(library(Matrix))
suppressMessages(library(viridis))
suppressMessages(library(cowplot))
suppressMessages(library(ggsci))
suppressMessages(library(reshape2))
library(arrow)
options(future.globals.maxSize= 891289600*16*1600)


getbin = function(slice_id){


readDir='/home/liuyx/liuyuxuan/spa_seurat/cla/macaca/renew_spatial_data_240129_macaque/240128_macaque_total_gene_2d/macaque-20240125-Mq179-cla/'
totalGeneFile=paste0(readDir,'total_gene_',slice_id,'_macaque_f001_2D_macaque-20240125.parquet')

message("Working cell summary on:", totalGeneFile)
#rtSBATCH --mem 1000000
totalGeneTb_liso = read_parquet(totalGeneFile)
#totalGeneTb_liso=totalGeneTb_liso[sample(nrow(totalGeneTb_liso),30000),]


annoDir=readDir
anno_area=read.csv(paste0(annoDir,'/region-macaque-20240125.csv'))
#area=unique(anno_area[anno_area$note %in% c('pu','cla','ins','amy'),]$global_region_id)
#totalGeneTb_liso=totalGeneTb_liso[totalGeneTb_liso$gene_area %in%area,]

#totalGeneTb_liso=totalGeneTb_liso[c(1:10000),]
totalGeneTb_liso$x=totalGeneTb_liso$rx
totalGeneTb_liso$y=totalGeneTb_liso$ry

totalGeneTb=totalGeneTb_liso

dnbSumTb = totalGeneTb %>% group_by(x, y) %>% summarize(
    umi_count = sum(umi_count),
    detectGeneCount = length(unique(gene)),
    cell_label = cell_label[1],
    gene_area = gene_area[1]
) %>% ungroup() # Be careful to ungroup before mutate

# For each cell, we use the gene_area with most DNB support.
cellSumTb = dnbSumTb %>% group_by(cell_label) %>% summarize(
    x = as.integer(mean(x)), # Use DNB average position as cell position
    y = as.integer(mean(y)),
    cell_area = ifelse(
        all(is.na(gene_area)), NA,
        names(sort(table(gene_area), decreasing=T))[1]
    )
) %>% ungroup()


# Remove cell_id 0
cellSumTb = subset(cellSumTb, cell_label != 0)
# cellSumFile = paste0(outCellSumDir, "/", sub("total_gene_", "total_cell_", f))
# write_tsv(cellSumTb, cellSumFile)

# Sum gene expression for each cell
cellCountTb = subset(totalGeneTb, cell_label != 0) %>% group_by(cell_label, gene) %>% summarize(
    umi_count = sum(umi_count)
) %>% ungroup()
cellCountTb = cellCountTb %>% mutate(
    gene = as.factor(gene),
    cell_label = as.factor(cell_label)
)

secId = "hip"
dimName = list(levels(cellCountTb$gene), levels(cellCountTb$cell_label))
cellCountMx = sparseMatrix(
    as.integer(cellCountTb$gene), as.integer(cellCountTb$cell_label),
    x=cellCountTb$umi_count, dimnames=dimName
)
cellBinSeurat = CreateSeuratObject(counts=cellCountMx, assay="Spatial")

coordinatesDf = cellSumTb[c("cell_label", "x", "y")] %>% tibble::column_to_rownames("cell_label")
colnames(coordinatesDf) = c("imagecol","imagerow")
coordinatesDf = coordinatesDf[rownames(cellBinSeurat@meta.data),]
addMetaDf = as.data.frame(cellSumTb[c(
    "cell_label", "x", "y", "cell_area"
)])
addMetaDf$secId = secId
rownames(addMetaDf) = addMetaDf$cell_label
addMetaDf = addMetaDf[rownames(cellBinSeurat@meta.data),]


#额外加区域信息
cell_id_label=addMetaDf$cell_label
colnames(anno_area)[7]='cell_area'
anno_area=anno_area[,c('origin_name','cell_area')]
anno_area=anno_area[!duplicated(anno_area),]
anno_area$cell_area=as.character(anno_area$cell_area)
addMetaDf=left_join(addMetaDf,anno_area)
if(nrow(addMetaDf[is.na(addMetaDf$cell_label),])>0){addMetaDf[is.na(addMetaDf$cell_label),]$cell_label=0}

#addMetaDf[is.na(addMetaDf$cell_label),]$cell_label=0
#rownames(addMetaDf)=cell_id_label

cellBinSeurat = AddMetaData(cellBinSeurat, addMetaDf$x,col.name = 'x')
cellBinSeurat = AddMetaData(cellBinSeurat, addMetaDf$y,col.name = 'y')
cellBinSeurat = AddMetaData(cellBinSeurat, addMetaDf$cell_area,col.name = 'cell_area')

#cellBinSeurat = AddMetaData(cellBinSeurat, metadata=addMetaDf)

MTGenes = c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
cellBinSeurat[["percent.mt"]] <- PercentageFeatureSet(cellBinSeurat,features = MTGenes)

outDir='/home/liuyx/liuyuxuan/spa_seurat/cla/macaca/renew_spatial_data_240129_macaque/cell_bin_boundary_179/'

saveRDS(cellBinSeurat, paste0(outDir,slice_id,'cellbin.rds'))

}

all_f=list.files('/home/liuyx/liuyuxuan/spa_seurat/cla/macaca/renew_spatial_data_240129_macaque/240128_macaque_total_gene_2d/macaque-20240125-Mq179-cla/')
allFile=all_f[grep('parquet',all_f)]
all_slice_id=str_extract(allFile,'T[0-9]*')
undone_id=all_slice_id
undone_id

thread = 4
mclapply(
    undone_id,
    function(slice_id) {getbin(slice_id)},
    mc.cores=thread
)



