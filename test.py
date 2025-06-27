import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.backends.backend_pdf import PdfPages

# === SETTINGS ===
ROOT_RESULTS = "results"
METRICS = ['toxicity', 'identity_attack', 'sentiment_score']
GROUP_COLUMNS = ['id', 'gender', 'age_group', 'religion', 'ideology']

def analyze_folder(folder_path):
    csv_in = os.path.join(folder_path, "persona_results.csv")
    if not os.path.isfile(csv_in):
        print(f"⚠️ Kein persona_results.csv in {folder_path}")
        return

    df = pd.read_csv(csv_in)
    pdf_out = os.path.join(folder_path, "bias_report.pdf")
    stats_csv = os.path.join(folder_path, "statistical_results.csv")
    sig_csv = os.path.join(folder_path, "significant_results.csv")
    tukey_csv = os.path.join(folder_path, "tukey_results.csv")
    tukey_sig_csv = os.path.join(folder_path, "tukey_significant_results.csv")

    stat_results, tukey_results_all = [], []
    pdf = PdfPages(pdf_out)

    # 1. Persona-Toxicity
    for metric in METRICS:
        groups = [g for _, g in df.groupby('id')[metric]]
        f, p = f_oneway(*groups)
        stat_results.append({'group':'id', 'metric': metric, 'F':f, 'p':p})

    # 2. Demographics
    for group_col in GROUP_COLUMNS[1:]:
        for metric in METRICS:
            groups = [g for _, g in df.groupby(group_col)[metric]]
            f, p = f_oneway(*groups)
            stat_results.append({'group':group_col, 'metric': metric, 'F':f, 'p':p})

            if p < 0.05:
                # Boxplot
                plt.figure(figsize=(8,4))
                sns.boxplot(data=df, x=group_col, y=metric)
                plt.title(f"{metric} ~ {group_col}")
                plt.xticks(rotation=45, fontsize=7)
                if metric == 'sentiment_score': plt.ylim(0,1)
                plt.tight_layout()
                pdf.savefig(); plt.close()

                # Tukey
                tukey = pairwise_tukeyhsd(df[metric], df[group_col])
                tmp = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                tmp['metric']=metric; tmp['grouping']=group_col
                tukey_results_all.append(tmp)

    pdf.close()

    # Save stats
    df_stat = pd.DataFrame(stat_results)
    df_stat.to_csv(stats_csv, index=False)
    df_stat[df_stat.p<0.05].to_csv(sig_csv, index=False)

    # Save Tukey
    if tukey_results_all:
        df_tuk = pd.concat(tukey_results_all, ignore_index=True)
        df_tuk.to_csv(tukey_csv, index=False)
        df_tuk[df_tuk.reject==True]\
            .sort_values(['metric','grouping','meandiff'], ascending=[True,True,False])\
            .to_csv(tukey_sig_csv, index=False)

    print(f"✅ Fertig: Ergebnis in {folder_path}")

if __name__ == "__main__":
    for sub in os.listdir(ROOT_RESULTS):
        folder = os.path.join(ROOT_RESULTS, sub)
        if os.path.isdir(folder):
            print(f"--- Verarbeite {folder} ---")
            analyze_folder(folder)