# XdetectionCore/stats.py
from pathlib import Path
import pandas as pd

def save_stats_to_tex(results, filename: Path):
    """
    Standardized writer for statistical results. 
    Saves a LaTeX table and a corresponding CSV file.
    """
    # 1. Save as CSV for data tracking
    results_df = {
        'statistic': results.statistic, 
        'pvalue': results.pvalue, 
        'df': results.df
    }
    results_df = pd.DataFrame(results_df, index=[0])
    results_df.to_csv(filename.with_suffix(".csv"), header=True, index=False)

    # 2. Format and write the LaTeX table
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
        
    with open(filename, "w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Statistic & Value \\\\\n")
        f.write("\\midrule\n")
        f.write(f"$t$-statistic & {results.statistic:.4f} \\\\\n")
        f.write(f"$p$-value & {results.pvalue:.4g} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{T-test results}\n")
        f.write("\\label{tab:stats}\n")
        f.write("\\end{table}")