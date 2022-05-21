import scanpy as sc, anndata
import pandas as pd, time, deepchem as dc

fda_fname = 'fda_zinc20.txt'
fda_url = 'https://zinc20.docking.org/substances/subsets/fda.txt:zinc_id+smiles+preferred_name+mwt+logp+rb?count=all'
fda_molecules = pd.read_csv(fda_url, header=None, index_col=0, sep='\t')
fda_molecules['dataset'] = 'FDA'

worldnotfda_fname = 'world-not-fda_zinc20.txt'
worldnotfda_url = 'https://zinc20.docking.org/substances/subsets/world-not-fda.txt:zinc_id+smiles+preferred_name+mwt+logp+rb?count=all'
worldnotfda_molecules = pd.read_csv(worldnotfda_url, header=None, index_col=0, sep='\t')
worldnotfda_molecules['dataset'] = 'world-not-FDA'

all_chems = pd.concat([fda_molecules, worldnotfda_molecules])
all_chems.columns = ['SMILES', 'Preferred name', 'Molecular weight', 'Log P', 'Rotatable bonds', 'dataset']

itime = time.time()
featurizer = dc.feat.CircularFingerprint()
feature_mat = featurizer.featurize(list(all_chems['SMILES']))
print(time.time() - itime)


# Calculating some chemical descriptors using RDKit.

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, Draw, rdMolDescriptors, Draw, PandasTools
from rdkit.Chem.Draw import IPythonConsole

featurizer = dc.feat.RDKitDescriptors()
features = featurizer.featurize(list(all_chems['SMILES']))
metadata_df = pd.DataFrame(data=features, index=all_chems.index, columns=featurizer.descriptors)

metadata_df = metadata_df.join(all_chems)
metadata_df.index.name = 'ZINC_ID'


# Calculating some chemical descriptors using RDKit.

anndata_all = anndata.AnnData(X=feature_mat, obs=metadata_df)

itime = time.time()
# Do PCA for denoising and dimensionality reduction
sc.pp.pca(anndata_all, n_comps=10)
print("PCA computed. Time: {}".format(time.time() - itime))

# Compute neighborhood graph in PCA space
sc.pp.neighbors(anndata_all)
print("Neighbors computed. Time: {}".format(time.time() - itime))

# Compute UMAP using neighborhood graph
sc.tl.umap(anndata_all)
print("UMAP calculated. Time: {}".format(time.time() - itime))

sc.pl.umap(anndata_all, color='Molecular weight') #, s=4)

anndata_all.obs['x'] = anndata_all.obsm['X_umap'][:, 0]
anndata_all.obs['y'] = anndata_all.obsm['X_umap'][:, 1]
anndata_all.write('approved_drugs.h5')
print("Data written. Time: {}".format(time.time() - itime))