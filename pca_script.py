import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def rolling_pca(data: pd.DataFrame, 
                nb_components: int=3, 
                w: int=126) -> pd.DataFrame:
    
    n = data.shape[0]
    prev_componets = None # We will need them for calculating dot product
    pcs_ls = [] # List to append last calculated values of PCs

    for t in range(w, n+1):
        curr_sample = data.iloc[(t-w):t, :]
        
        # Find PCs
        curr_pca = PCA(n_components=nb_components)
        curr_z = curr_pca.fit_transform(curr_sample)

        # Mitigate sign instability
        if prev_componets is not None: # In the first iteration, there are no previous components
            for j in range(0, nb_components):
                if np.dot(curr_pca.components_[j], prev_componets[j]) < 0:
                    # Change the sign if it flipped unexpectedly
                    curr_pca.components_[j] *= -1
                    curr_z[:, j] *= -1

        # Update eigenvectors
        prev_componets = curr_pca.components_.copy()
        # Find dataframe of PCs
        curr_z_df = pd.DataFrame(curr_z, columns=[f'PC{i}' for i in range(1, nb_components+1)], index=curr_sample.index)
        # Add last observation as dataframe, not series
        pcs_ls.append(curr_z_df.iloc[-2:-1, :])
    
    # Concatenate results
    all_pcs = pd.concat(pcs_ls, axis=0)

    return all_pcs

