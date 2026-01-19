import pandas as pd
import numpy as np
from pathlib import Path

def convert_runtimes_to_latex(csv_path, output_path=None, q_percent=0.1):
    """
    Convert runtime CSV to LaTeX table with colored formatting.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the LaTeX file (optional)
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Convert q proportions to percentages
    if q_percent == 0.1: 
        q_percent = 10
    if q_percent == 0.25: 
        q_percent = 25
    if q_percent == 0.5: 
        q_percent = 50
    
    # Define colors for methods (using ggsci NPG palette)
    method_colors = {
        'ArgMax Product': 'npgRed',
        'Independent': 'npgPurple',
        'Max Product': 'npgBlue',
        'PAC_MAP': 'npgGreen',
        'PAC_MAP_Hamming': 'npgNavy'
    }
    
    # Parse mean and std from "mean+/-std" format and find minimum per row
    formatted_data = []
    for _, row in df.iterrows():
        dataset = row['Dataset']
        row_data = {'Dataset': dataset}
        
        # Extract means for comparison
        means = {}
        for method in method_colors.keys():
            if method in df.columns:
                val_str = row[method]
                mean_val = float(val_str.split('+/-')[0])
                means[method] = mean_val
        
        # Find minimum mean (fastest method)
        min_mean = min(means.values())
        
        # Format each method's runtime
        for method in method_colors.keys():
            if method in df.columns:
                val_str = row[method]
                mean_str, std_str = val_str.split('+/-')
                mean_val = float(mean_str)
                
                color = method_colors[method]
                
                # Bold the fastest method
                if mean_val == min_mean:
                    formatted = f"$\\textcolor{{{color}}}{{\\bm{{{mean_str}}}}} \\pm {std_str}$"
                else:
                    formatted = f"$\\textcolor{{{color}}}{{{mean_str}}} \\pm {std_str}$"
                
                row_data[method] = formatted
        
        formatted_data.append(row_data)
    
    df_formatted = pd.DataFrame(formatted_data)
    
    # Build LaTeX table
    latex_str = r"% Requires \usepackage{xcolor}, \usepackage{booktabs}, \usepackage{bm}" + "\n"
    latex_str += r"% Define ggsci NPG palette colors" + "\n"
    latex_str += r"\definecolor{npgRed}{HTML}{E64B35}" + "\n"
    latex_str += r"\definecolor{npgBlue}{HTML}{4DBBD5}" + "\n"
    latex_str += r"\definecolor{npgGreen}{HTML}{00A087}" + "\n"
    latex_str += r"\definecolor{npgNavy}{HTML}{3C5488}" + "\n"
    latex_str += r"\definecolor{npgPurple}{HTML}{8491B4}" + "\n"
    latex_str += "\n"
    
    latex_str += r"\begin{table}[ht]" + "\n"
    latex_str += r"\centering" + "\n"
    latex_str += r"\scriptsize" + "\n"
    latex_str += r"\caption{Runtime comparison (seconds) for " f"{q_percent}" r"\% query size. We report the mean $\pm$ SE over ten trials.}" + "\n"
    latex_str += r"\label{tab:runtimes}" + "\n"
    latex_str += r"\begin{tabular}{lrrrrr}" + "\n"
    latex_str += r"\toprule" + "\n"
    
    # Header row
    latex_str += "Dataset & AMP & IND & MP & PAC-MAP & smooth-PAC-MAP \\\\\n"
    latex_str += r"\midrule" + "\n"
    
    # Data rows
    for _, row in df_formatted.iterrows():
        dataset = row['Dataset']
        
        # Handle special dataset names
        if dataset == "pumsb_star":
            dataset_str = r"\texttt{pumsb\_star}"
        elif dataset == "ocr_letters":
            dataset_str = r"\texttt{ocr\_letters}"
        else:
            dataset_str = r"\texttt{" + dataset + "}"
        
        values = [
            row['ArgMax Product'],
            row['Independent'],
            row['Max Product'],
            row['PAC_MAP'],
            row['PAC_MAP_Hamming']
        ]
        
        latex_str += dataset_str + " & " + " & ".join(values) + r" \\" + "\n"
    
    latex_str += r"\bottomrule" + "\n"
    latex_str += r"\end{tabular}" + "\n"
    
    # Add legend - actually don't think we need since each column is labelled 
    # with the method name
    # latex_str += r"\begin{center}" + "\n"
    # latex_str += r"\footnotesize" + "\n"
    # latex_str += r"\textcolor{npgRed}{\texttt{AMP}} \quad " + "\n"
    # latex_str += r"\textcolor{npgPurple}{\texttt{IND}} \quad " + "\n"
    # latex_str += r"\textcolor{npgBlue}{\texttt{MP}} \quad " + "\n"
    # latex_str += r"\textcolor{npgGreen}{\texttt{PAC-MAP}} \quad " + "\n"
    # latex_str += r"\textcolor{npgNavy}{\texttt{smooth-PAC-MAP}}" + "\n"
    # latex_str += r"\end{center}" + "\n"
    latex_str += r"\end{table}" + "\n"
    
    # Print to console
    print(latex_str)
    
    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"\nLaTeX table saved to {output_path}")
    
    return latex_str

def convert_runtimes_to_latex_simple(csv_path, output_path=None):
    """
    Convert runtime CSV to simple LaTeX table.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the LaTeX file (optional)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Find minimum runtime for each dataset
    formatted_data = []
    for _, row in df.iterrows():
        dataset = row['Dataset']
        row_data = {'Dataset': dataset}
        
        # Extract means
        means = {}
        for col in df.columns[1:]:  # Skip 'Dataset' column
            val_str = row[col]
            mean_val = float(val_str.split('+/-')[0])
            means[col] = mean_val
        
        min_mean = min(means.values())
        
        # Format with bold for minimum
        for col in df.columns[1:]:
            val_str = row[col]
            mean_val = float(val_str.split('+/-')[0])
            
            if mean_val == min_mean:
                row_data[col] = f"\\textbf{{{val_str}}}"
            else:
                row_data[col] = val_str
        
        formatted_data.append(row_data)
    
    df_formatted = pd.DataFrame(formatted_data)
    
    # Convert to LaTeX
    latex_str = df_formatted.to_latex(
        index=False,
        escape=False,
        column_format='lccccc',
        caption='Runtime comparison (seconds) for 10\\% query size',
        label='tab:runtimes'
    )
    
    # Replace column names with shorter versions
    latex_str = latex_str.replace('ArgMax Product', 'ArgMax')
    latex_str = latex_str.replace('Max Product', 'Max')
    latex_str = latex_str.replace('PAC\\_MAP\\_Hamming', 'Smooth')
    latex_str = latex_str.replace('PAC\\_MAP', 'PAC-MAP')
    
    # Add booktabs
    latex_str = latex_str.replace('\\hline', '\\midrule')
    latex_str = latex_str.replace('\\midrule\n\\midrule', '\\midrule')
    
    print(latex_str)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"\nLaTeX table saved to {output_path}")
    
    return latex_str


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MAP Benchmark Experiment Params')
    parser.add_argument("-id", "--id", default="06_01_26_3",
                        help="ID of experiment to plot runtimes from")
    parser.add_argument('-q', '--q-percent', type=float, required=True,
                        help="Proportion of query variables")
    parser.add_argument('-e', '--e-percent', type=float, required=True,
                        help="Proportion of evidence variables")
    args = parser.parse_args()
    exp_id = args.id
    q_percent = args.q_percent
    e_percent = args.e_percent

    # Example usage
    csv_path = Path(f"results/{exp_id}/20-datasets_runtimes_{q_percent}q{e_percent}e_{exp_id}.csv")
    output_path = Path(f"results/{exp_id}/runtimes_table.tex")
    
    latex_table = convert_runtimes_to_latex(csv_path, output_path, q_percent)