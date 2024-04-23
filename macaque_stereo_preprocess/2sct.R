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
suppressMessages(library(stringr))

options(future.globals.maxSize= 891289600*20000000000000000000000000)

workDir='/home/liuyx/liuyuxuan/spa_seurat/cla/macaca/renew_spatial_data_240129_macaque/cell_bin_boundary_179/'
alls=list.files(paste0(workDir))

slice=str_extract(alls,'T[0-9]*')
slice=slice[!(is.na(slice))]
done_file=list.files('/home/liuyx/liuyuxuan/spa_seurat/cla/macaca/renew_spatial_data_240129_macaque/cell_bin_boundary_179_sct')

done_slice=done_file[grep('rds$',done_file)]
done_slice=str_extract(done_slice,'T[0-9]*')


undone_slice=setdiff(slice,done_slice)
undone_slice


mclapply(
    'T39',
    function(secId) {secCellSeurat=readRDS(paste0('/home/liuyx/liuyuxuan/spa_seurat/cla/macaca/renew_spatial_data_240129_macaque/cell_bin_boundary_179/',secId,'cellbin.rds'))
secCellSeurat$area_name_mod=secCellSeurat$cell_area
secCellSeurat$cell_id=colnames(secCellSeurat)
secCellSeurat$area_name_mod = ifelse(is.na(secCellSeurat$area_name_mod), "NotSign", secCellSeurat$area_name_mod)
MTGenes = c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
secCellSeurat[["percent.mt"]] <- PercentageFeatureSet(secCellSeurat,features = MTGenes)
secCellSeurat = subset(secCellSeurat, cell_id != 0) # cell_id == 0 means not signed to a cell
secCellSeuratFilt = subset(secCellSeurat, area_name_mod != "NotSign" & cell_id != 0)
#secCellSeuratFilt = subset(secCellSeuratFilt, nFeature_Spatial>100 & percent.mt<15)
filtRatio = nrow(secCellSeuratFilt@meta.data) / nrow(secCellSeurat@meta.data)
secCellSeuratFilt = SCTransform(
            secCellSeuratFilt, ncells=ncol(secCellSeuratFilt[["Spatial"]]),
            variable.features.n=nrow(secCellSeuratFilt[["Spatial"]]),
            assay="Spatial",
            vars.to.regress="percent.mt",
            method="glmGamPoi",
            verbose=F
        )
#             RunPCA(npcs=100, assay = "SCT", verbose=F) %>%
#             FindNeighbors(reduction = "pca", dims = 1:50, verbose=F)  %>%
#             FindClusters(verbose=F) %>%
#             RunUMAP(reduction = "pca", dims = 1:50, verbose=F)
saveRDS(secCellSeuratFilt, file=paste0('/home/liuyx/liuyuxuan/spa_seurat/cla/macaca/renew_spatial_data_240129_macaque/cell_bin_boundary_179_sct/', secId, '_cla_sct.rds'))
print(paste0( " Filter ratio: ", filtRatio))
                    },
    mc.cores=2
)

