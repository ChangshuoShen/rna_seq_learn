import scanpy as sc
import pandas as pd
import numpy as np
import os
import shutil

class scRNAseqUtils:
    def __init__(self, adata, working_dir):
        """
        初始化类时，传入 AnnData 对象（scanpy 的核心数据结构）和工作目录
        """
        self.adata = adata
        self.working_dir = working_dir
        # 确保工作目录存在
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        
        # 计算 QC 指标
        self.calculate_qc_metrics()

    def calculate_qc_metrics(self):
        """
        计算质量控制所需的指标，如总 counts、基因数和线粒体基因比例
        """
        # 标记线粒体基因
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        
        # 计算质量控制指标
        sc.pp.calculate_qc_metrics(
            self.adata, 
            qc_vars=['mt'], 
            percent_top=None, 
            log1p=False, 
            inplace=True)
        
    def filter_cells_and_genes(
        self, min_genes=200, min_cells=3, max_mito=0.1):
        """
        过滤基因和细胞：
            - 过滤在少于min_cells个细胞中表达的基因。
            - 过滤表达低于min_genes或高于max_genes的细胞，以及线粒体基因表达高于max_mito的细胞。
        """
        # 过滤之前可视化一遍
        self.violin_plots('violin_before.png')
        self.plot_highest_expr_genes('highest_expr_genes_before.png')
        
        sc.pp.filter_genes(self.adata, min_cells=min_cells)# 基于细胞数过滤基因
        sc.pp.filter_cells(self.adata, min_genes=min_genes)# 过滤基于基因表达数的细胞
        
        # 过滤线粒体基因表达比例过高的细胞，暂时注释
        # self.adata = self.adata[self.adata.obs['pct_counts_mt'] < max_mito, :]
        
        # 更新 QC 指标
        self.calculate_qc_metrics()
        # 过滤之后可视化一遍
        self.violin_plots('violin_after.png')
        self.plot_highest_expr_genes('highest_expr_genes_after.png')

    def violin_plots(self, save_name):
        """
        使用小提琴图可视化基因数、UMI counts 和线粒体基因比例，并保存图片。
        """
        default_save_path = 'figures/violin.png'
        save_path = os.path.join(self.working_dir, save_name)
        sc.pl.violin(
            self.adata, 
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
            jitter=0.4, 
            multi_panel=True,
            save='.png',
            show=False
        )
        # 移动图像到工作目录
        shutil.move(default_save_path, save_path)
        return save_path

    def plot_highest_expr_genes(self, save_name, n_top_genes=20):
        """
        可视化表达最高的前 n 个基因，并保存图片。
        """
        default_save_path = 'figures/highest_expr_genes.png'
        save_path = os.path.join(
            self.working_dir, save_name
        )
        sc.pl.highest_expr_genes(
            self.adata, 
            n_top=n_top_genes, 
            save='.png',
            show=False
        )
        shutil.move(default_save_path, save_path)
        return save_path

    def normalize_data(self):
        """
        对数据进行标准化处理，每个细胞归一化到相同的总表达量，并对数转换。
        """
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)


    def find_highly_variable_genes(self, save_name):
        """
        找出高变异基因，用于后续分析。
        """
        default_save_path = 'figures/filter_genes_dispersion.png'
        save_path = os.path.join(self.working_dir, save_name )
        sc.pp.highly_variable_genes(
            self.adata, 
            min_mean=0.0125, 
            max_mean=3, 
            min_disp=0.5)
        sc.pl.highly_variable_genes(
            self.adata, 
            save='.png',
            show=False
        )
        shutil.move(default_save_path, save_path)
        return save_path

    def filter_hvg(self):
        """
        选择高变异基因并保留原始数据。
        该方法只能执行一次，因为它会修改 adata 对象。
        """
        self.adata.raw = self.adata  # 保留原始数据
        self.adata = self.adata[:, self.adata.var['highly_variable']]  # 选择高变异基因
        print(self.adata.raw.shape, self.adata.shape)
    
    def scale_data(self):
        """
        对数据进行标准化，使得每个基因的表达量具有均值为0、方差为1。
        """
        # 将数据缩放到单位方差
        sc.pp.regress_out(
            self.adata, 
            ['total_counts', 'pct_counts_mt']# 要回归出去的变量列表
        )
        sc.pp.scale(self.adata, max_value=10)

    def pca(self, n_comps=50):
        """
        进行主成分分析(PCA)，并将结果保存在adata对象中。
        """
        default_save_path = 'figures/pca.png'
        save_path = [
            os.path.join(self.working_dir, 'pca.png'),
            os.path.join(self.working_dir, 'pca_variance_ratio.png')
        ]
        sc.tl.pca(
            self.adata, 
            # n_comps=n_comps,
            svd_solver='arpack' # SVD求解器，'arpack' 适用于大型稀疏矩阵
            )
        sc.pl.pca(
            self.adata, 
            color='CST3', 
            save='.png',
            show=False
        )
        sc.pl.pca_variance_ratio(
            self.adata, 
            log=True, 
            show=False,
            save='.png')
        
        shutil.move('figures/pca.png', save_path[0])
        shutil.move('figures/pca_variance_ratio.png', save_path[1])
        return save_path
    def neighbors(self, n_neighbors=10, n_pcs=40):
        """
        计算邻居，用于后续降维（如UMAP和t-SNE）和聚类。
        """
        sc.pp.neighbors(
            self.adata, 
            n_neighbors=n_neighbors, 
            n_pcs=n_pcs)
        
        
    def tsne(self):
        """
        使用 t-SNE 方法对数据进行降维。
        """
        sc.tl.tsne(self.adata)
        sc.pl.tsne(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'tsne.png')
        shutil.move('figures/tsne.png', save_path)
        return save_path

    def umap(self):
        """
        使用 UMAP 方法对数据进行降维。
        """
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap.png')
        shutil.move('figures/umap.png', save_path)
        return save_path

    def leiden(self, resolution=1.0):
        '''
        使用Leiden聚类
        '''
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.pl.umap(
            self.adata, 
            color=['leiden'], 
            save='_leiden.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_leiden.png')
        shutil.move('figures/umap_leiden.png', save_path)
        return save_path
    
    def louvain(self, resolution=1.0):
        '''
        使用Louvain聚类
        '''
        # 计算邻近（如果尚未计算）
        # sc.pp.neighbors(self.adata, use_rep='X_pca')

        # 进行Louvain聚类
        sc.tl.louvain(self.adata, resolution=resolution)

        # 可视化结果
        sc.pl.umap(
            self.adata, 
            color=['louvain'], 
            save='_louvain.png',
            show=False
        )
        
        # 移动保存的文件
        save_path = os.path.join(self.working_dir, 'umap_louvain.png')
        shutil.move('figures/umap_louvain.png', save_path)
        
        return save_path

# 测试一下
if __name__ == '__main__':
    
    # 定义 AnnData 对象
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    # 定义工作目录
    working_dir = './qc_plots'

    # 初始化 scRNAseqUtils 类
    utils = scRNAseqUtils(adata, working_dir)

    # 测试过滤函数
    utils.filter_cells_and_genes()
    
    utils.normalize_data()
    utils.find_highly_variable_genes('HVG.png')
    utils.filter_hvg()
    utils.scale_data()
    utils.neighbors()
    utils.pca()
    utils.tsne()
    utils.umap()
    utils.leiden()
    utils.louvain()

