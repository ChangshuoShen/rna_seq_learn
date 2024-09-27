import scanpy as sc
import pandas as pd

class DataProcessor:
    def __init__(self, file_path, file_type='h5ad'):
        self.file_path = file_path
        self.file_type = file_type
        self.adata = None

    def load_data(self):
        '''
        根据文件类型加载数据
        '''
        if self.file_type == 'h5ad':
            self.adata = sc.read_h5ad(self.file_path)
        elif self.file_type == 'csv':
            df = pd.read_csv(self.file_path, index_col=0)
            self.adata = sc.AnnData(df)
        elif self.file_type == 'xlsx':
            df = pd.read_excel(self.file_path, index_col=0)
            self.adata = sc.AnnData(df)
        else:
            raise ValueError("Unsupported file type. Please use 'h5ad', 'csv', or 'xlsx'.")

    def preprocess_data(self):
        '''
        数据预处理，包含质量控制、标准化等
        '''
        # 质量控制
        sc.pp.filter_cells(self.adata, min_genes=200)
        sc.pp.filter_genes(self.adata, min_cells=3)
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        self.adata = self.adata[self.adata.obs.n_genes_by_counts < 2500, :]
        self.adata = self.adata[self.adata.obs.pct_counts_mt < 5, :]

        # 数据标准化
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

        # 特征选取
        sc.pp.highly_variable_genes(self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        self.adata.raw = self.adata
        self.adata = self.adata[:, self.adata.var.highly_variable]
        sc.pp.regress_out(self.adata, ['total_counts', 'pct_counts_mt'])
        sc.pp.scale(self.adata, max_value=10)

    def run_analysis(self):
        '''
        运行聚类和可视化
        '''
        sc.pp.neighbors(self.adata, n_neighbors=10)
        sc.tl.umap(self.adata)
        sc.tl.leiden(self.adata)
        sc.pl.umap(self.adata, color=['leiden'], save='umap_leiden.png')

    def find_marker_genes(self):
        '''
        发现标志基因
        '''
        sc.tl.rank_genes_groups(self.adata, 'leiden', method='t-test')
        sc.pl.rank_genes_groups(self.adata, n_genes=25, sharey=False)

        # 细胞类型标注
        new_cluster_names = [
            'CD4 T',
            'CD14 Monocytes',
            'B',
            'CD8 T',
            'FCGR3A Monocytes',
            'NK',
            'Dendritic',
            'Megakaryocytes']
        self.adata.rename_categories('leiden', new_cluster_names)
        sc.pl.umap(
            self.adata, 
            color='leiden', 
            legend_loc='on data', 
            title='', 
            frameon=False, 
            save='_pbmc3k_annotation.png')
        
if __name__ == '__main__':
    
    # 使用示例
    file_path = 'data/pbmc3k_raw.h5ad'  # 或其他文件路径
    data_processor = DataProcessor(file_path, file_type='h5ad')
    data_processor.load_data()
    data_processor.preprocess_data()
    data_processor.run_analysis()
    data_processor.find_marker_genes()
