#### Principal factors
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Read in subset of data.
cds_data = pd.read_excel("./Data/subset_data.xlsx")

for firm in ['CMZB','DANBNK','MONTE','SVSKHB']:
    test_df = cds_data[(cds_data['Ticker']==firm)]
    test_df = test_df.pivot(index = ['Date','Ticker'],
                            columns='Tenor',values = 'Par Spread').reset_index()
    # Test on subset data ownly to get very few obs. One large spread increase to test.
    test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

    t = np.array(test_df['Years'])

    maturities = np.array([1,3,5,7,10])
    maturities = np.array([1,2,3,4,5,7,10,15,20,30])
    t_mat_grid = np.ascontiguousarray(maturities[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
    # Forward fill in case of nans.
    CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y',
                                '15Y','20Y','30Y']].ffill().bfill())
    # CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

    # PCA. Standardize obs.
    scaler = StandardScaler()
    cds_scaled = scaler.fit_transform(CDS_obs)

    print("--- Data Standardization Complete ---\n")


    # --- 3. PERFORM PRINCIPAL COMPONENT ANALYSIS (PCA) ---
    # We calculate up to the maximum number of components (5).
    pca = PCA(n_components=len(maturities))
    pca.fit(cds_scaled)


    # --- 4. RESULTS AND INTERPRETATION ---

    # Explained Variance Ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("--- Explained Variance Ratio per Principal Component ---")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC {i+1}: {ratio * 100:.2f}%")
    print(f"Cumulative Explained Variance: {explained_variance_ratio.cumsum()[-1]*100:.2f}%\n")


    # Components/Loadings (Interpretation of Factors)
    # The sign of the components is arbitrary, but the pattern is not.
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(len(maturities))], 
        index=maturities
    )

    print("--- Principal Component Loadings (Weights) ---")
    print("These indicate the composition of each factor:")
    print(loadings.round(4))
    print("\nInterpretation:")
    print(f"PC1 (Level): All loadings are similar and positive. It represents an overall shift in the credit curve.")
    print(f"PC2 (Slope): Loadings show opposite signs (e.g., negative short-end, positive long-end). It represents the tilt/steepness of the curve.")


# Plot  
    plt.figure(figsize=(10,6))

    plt.plot(loadings["PC1"], marker='o', label="PC1 (Level)")
    plt.plot(loadings["PC2"], marker='o', label="PC2 (Slope)")
    plt.plot(loadings["PC3"], marker='o', label="PC3 (Curvature / Hump)")

    plt.xlabel("Maturity")
    plt.ylabel("Loading")
    plt.title("First Three Principal Components vs Maturity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Results/PCA/Components_{firm}.png')
    plt.close()

    # --- 5. VISUALIZATION (Scree Plot) ---
    plt.figure(figsize=(10, 6))

    # Plot explained variance
    plt.plot(range(1, len(maturities) + 1), explained_variance_ratio.cumsum(), 
            marker='o', linestyle='--', color='blue', label='Cumulative Variance')

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
    plt.title('Explained Variance by Principal Component', fontsize=16)
    plt.xlabel('Principal Component Number', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.xticks(range(1, len(maturities) + 1))
    plt.grid(axis='y', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Results/PCA/ExplainedVariance_{firm}.png')
    plt.close()

    # --- 6. TRANSFORM DATA (Optional) ---
    # Get the time series of the principal components (the factors themselves)
    cds_factors = pca.transform(cds_scaled)
    cds_factors_df = pd.DataFrame(
        cds_factors, 
        columns=[f'PC{i+1}' for i in range(len(maturities))]
    )
    print("\n--- First 5 rows of CDS Factors (Time Series) ---")
    print(cds_factors_df.head())
    print(f"\nIf you were to use a 3-factor model (as suggested by many credit models):")
    print(f"The first 3 factors explain {cds_factors_df.iloc[:, :3].var(axis=0).sum() / cds_factors_df.var(axis=0).sum() * 100:.2f}% of the total variance.")


    # --- 7. VISUALIZATION (Factor Time Series Plot) ---
    plt.figure(figsize=(12, 8))
    plt.plot(cds_factors_df['PC1'], label='PC1 (Level)', linewidth=2)
    plt.plot(cds_factors_df['PC2'], label='PC2 (Slope)', linewidth=2)
    if len(maturities) >= 3:
        plt.plot(cds_factors_df['PC3'], label='PC3 (Curvature)', linewidth=2)

    plt.title('Time Series of the First Three Principal Components', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Component Score (Standardized)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'./Results/PCA/Factors_{firm}.png')
    plt.close()

etst = 1

