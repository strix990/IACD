### For the analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

odf = pd.read_excel("./DATASET.xls")
#odf = pd.read_excel("./dataset_limpo.xlsx")

def Histogram():
    df = odf.drop("ID", axis=1)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        print("No numeric columns found.")
        return
    num_df.hist(figsize=(12, 10), bins=30, edgecolor='black')
    plt.suptitle("Histograms for All Numeric Attributes", fontsize=16)
    plt.tight_layout()
    plt.show()

def BoxPlot():
   
    cols_to_plot = ["Peso", "Altura", "IMC", "PA SISTOLICA", "PA DIASTOLICA"]
    df = odf[cols_to_plot]
    
    num_cols = len(df.columns)
    # Calculate layout: 3 columns, with just enough rows
    n_rows = (num_cols + 2) // 3 
    
    # Create the figure and subplots
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(15, 4 * n_rows), 
        sharex=False, sharey=False
    )
    axes = axes.flatten()  # Flattens the axes array for easy iteration

    # Iterate and plot, applying styling for a better look
    for i, col in enumerate(df.columns):
        ax = axes[i]
        
        # Core plotting with customizations
        bp = df.boxplot(
            column=col, ax=ax, patch_artist=True, 
            return_type='dict', medianprops={'color': 'red', 'linewidth': 2}
        )
        
        # Apply color to the box
        for box in bp['boxes']:
            box.set(facecolor='teal', edgecolor='darkblue', alpha=0.7)
            
        # Customize outliers (fliers)
        for flier in bp['fliers']:
            flier.set(marker='o', color='red', alpha=0.6, markersize=6)
            
        ax.set_title(col, fontsize=14, fontweight='bold')
        ax.set_ylabel("Values", fontsize=11)
        ax.tick_params(axis='x', labelbottom=False) # Remove x-axis labels
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Remove any unused subplots
    for j in range(num_cols, len(axes)):
        fig.delaxes(axes[j])

    # Final layout adjustments
    plt.suptitle("📊 Boxplots for All Numeric Attributes", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    plt.show()

def SpreadMeasure():
    df = odf.drop("ID", axis=1)
    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        print("No numeric columns found.")
        return
    desc = num_df.describe().T
    desc["range"] = desc["max"] - desc["min"]
    print("\nSpread Measures:\n")
    print(desc[["mean", "std", "min", "25%", "50%", "75%", "max", "range"]])

def DensityPlot():
    df = odf.drop("ID", axis=1)
    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        print("No numeric columns found.")
        return
    num_df.plot(kind='density', subplots=True, layout=(int(len(num_df.columns)/3)+1, 3),
                figsize=(12, 10), sharex=False)
    plt.suptitle("Density Plots for All Numeric Attributes", fontsize=16)
    plt.tight_layout()
    plt.show()

#SpreadMeasure()
#DensityPlot()
BoxPlot()
#Histogram()