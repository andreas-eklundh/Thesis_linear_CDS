#### Principal factors
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Read in subset of data.
cds_data = pd.read_excel("./Data/subset_data.xlsx")

for firm in ['DANBNK','MONTE']:
    test_df = cds_data[(cds_data['Ticker']==firm)]
    test_df = test_df.pivot(index = ['Date','Ticker'],
                            columns='Tenor',values = 'Par Spread').reset_index()
    # Test on subset data ownly to get very few obs. One large spread increase to test.
    test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

    t = np.array(test_df['Years'])

    # maturities = np.array([1,3,5,7,10])
    maturities = np.array([1,2,3,4,5,7,10])
    t_mat_grid = np.ascontiguousarray(maturities[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
    # Forward fill in case of nans.
    CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y']].ffill().bfill())
    # CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

    # PCA. Scale to unit variance and mean zero. Good practice when Euckl. work. Also, var likeli different 
    scaler = StandardScaler()
    
    cds_scaled = scaler.fit_transform(CDS_obs)

    # PCA: At most number of maturity components (not logical as we want to restrict.)
    # 
    pca = PCA(n_components=len(maturities))
    pca.fit(cds_scaled)

    # Explained Variance Ratio
    explained_variance_ratio = pca.explained_variance_ratio_


    # Components/Loadings (Interpretation of Factors)
    # The sign of the components is arbitrary, but the pattern is not.
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(len(maturities))], 
        index=maturities
    )


    plt.figure(figsize=(10, 6))

    # Plot individual variance bars
    bars = plt.bar(range(1, len(maturities) + 1), explained_variance_ratio, 
                alpha=0.6, color='skyblue', label='Individual Variance')

    # Add text labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",               # value rounded to 3 decimals
            ha='center', va='bottom', 
            fontsize=9
        )

    # Labels and title
    plt.xlabel('Principal Component Number', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.xticks(range(1, len(maturities) + 1))
    plt.grid(axis='y', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Results/PCA/ExplainedVariance_{firm}.png')
    plt.close()

etst = 1

