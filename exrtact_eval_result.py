"""
Script to extract evaluation results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics():
    results_dir = Path('results')
    output_dir = Path('tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        'gpt-4o-mini', 'llama-4', 'qwen3-235b', 'gpt-oss-120b',
        'gpt-5.2', 'gemini-3-flash', 'claude-sonnet-4.5', 'qwen3-max'
    ]
    model_map = {
        'gpt-4o-mini': 'GPT-4o-mini',
        'llama-4': 'LLaMA-4',
        'qwen3-235b': 'Qwen3-235b',
        'gpt-oss-120b': 'GPT-OSS-120b',
        'gpt-5.2': 'GPT-5.2',
        'gemini-3-flash': 'Gemini-3-flash',
        'claude-sonnet-4.5': 'Claude-sonnet-4.5',
        'qwen3-max': 'Qwen3-Max'
    }
    tasks = ['task1', 'task2', 'task3', 'task4', 'task5']
    
    acc_data = []
    f1_data = []
    
    for model in models:
        model_acc = {'Model': model_map.get(model, model)}
        model_f1 = {'Model': model_map.get(model, model)}
        
        for task in tasks:
            filepath = results_dir / task / '0shot_cot' / f'{model}_{task}.jsonl'
            if not filepath.exists():
                print(f"Warning: {filepath} not found.")
                model_acc[task] = np.nan
                model_f1[task] = np.nan
                continue
                
            y_true = []
            y_pred = []
            judges = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    ans = data.get('answer')
                    prd = data.get('pred')
                    jdg = data.get('judge')
                    
                    if ans is None or prd is None:
                        continue
                    
                    y_true.append(ans)
                    y_pred.append(prd)
                    judges.append(jdg)
            
            if judges:
                acc = np.mean([1 if j else 0 for j in judges])
                f1 = f1_score(y_true, y_pred, average='macro')
                model_acc[task] = round(acc * 100, 2)
                model_f1[task] = round(f1, 4)
            else:
                model_acc[task] = np.nan
                model_f1[task] = np.nan
        
        acc_data.append(model_acc)
        f1_data.append(model_f1)
    
    # Create DataFrames
    df_acc = pd.DataFrame(acc_data)
    df_f1 = pd.DataFrame(f1_data)
    
    # Save CSV
    df_acc.to_csv(output_dir / 'table1_acc.csv', index=False)
    df_f1.to_csv(output_dir / 'table1_f1.csv', index=False)
    
    # Custom LaTeX generation for styled table
    def generate_styled_latex(df, output_path, is_acc=True):
        header_map = {
            'Model': 'Model',
            'task1': 'Phage\\\\Identification',
            'task2': 'Contamination\\\\Detection',
            'task3': 'Completeness\\\\(Orig)',
            'task4': 'Lifestyle\\\\Classification',
            'task5': 'Taxonomic\\\\Classification'
        }
        
        # Calculate Avg.
        numeric_cols = [c for c in df.columns if c != 'Model']
        df['Avg.'] = df[numeric_cols].mean(axis=1)
        
        # Find best per column
        best_vals = df[numeric_cols + ['Avg.']].max()
        
        latex_lines = []
        latex_lines.append(r"\begin{table*}[]")
        latex_lines.append(r"\centering")
        metric_name = "accuracy (%)" if is_acc else "F1 score"
        latex_lines.append(r"\caption{Model performance " + metric_name + r" on PhageBench tasks using 0shot-cot mode. Avg. represents the average across all tasks. The best performance in each column is bolded.}")
        latex_lines.append(r"\resizebox{\textwidth}{!}{%")
        num_cols = len(df.columns)
        col_spec = "l" + "c" * (num_cols - 1)
        latex_lines.append(r"\begin{tabular}{" + col_spec + r"}")
        latex_lines.append(r"\toprule")
        
        # Headers
        headers = []
        for col in df.columns:
            label = header_map.get(col, col)
            headers.append(r"\textbf{\begin{tabular}[c]{@{}c@{}}" + label + r"\end{tabular}}")
        latex_lines.append(" & ".join(headers) + r" \\ \midrule")
        
        latex_lines.append(f"\\multicolumn{{{num_cols}}}{{c}}{{\\cellcolor[HTML]{{E9EEEA}}\\textbf{{PhageBench}}}} \\\\ \\midrule")
        
        # Data rows
        for _, row in df.iterrows():
            line_cells = [str(row['Model'])]
            for col in numeric_cols + ['Avg.']:
                val = row[col]
                fmt = "{:.2f}" if is_acc else "{:.4f}"
                val_str = fmt.format(val)
                if val == best_vals[col]:
                    val_str = r"\textbf{" + val_str + "}"
                line_cells.append(val_str)
            latex_lines.append(" & ".join(line_cells) + r" \\")
            
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}%")
        latex_lines.append(r"}")
        latex_lines.append(r"\label{tab:phagebench_" + ("acc" if is_acc else "f1") + r"}")
        latex_lines.append(r"\end{table*}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(latex_lines))

    generate_styled_latex(df_acc.copy(), output_dir / 'table1_acc.tex', is_acc=True)
    generate_styled_latex(df_f1.copy(), output_dir / 'table1_f1.tex', is_acc=False)
    
    print(f"Tables saved to {output_dir}")

if __name__ == '__main__':
    calculate_metrics()
