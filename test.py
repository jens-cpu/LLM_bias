import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, mannwhitneyu, levene, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np

# === SETTINGS ===
ROOT_RESULTS = "results"
METRICS = ['toxicity', 'identity_attack', 'sentiment_score']
GROUP_COLUMNS = ['gender', 'age_group', 'religion', 'ideology']

# === Hilfsfunktion: Statistische Methode basierend auf Tests wählen ===
def check_distribution_and_select_test(df, group_col, metric):
    group_values = [g[metric].values for _, g in df.groupby(group_col)]
    if len(group_values) == 2:
        return 'mannwhitney'

    normal = True
    for _, values in df.groupby(group_col)[metric]:
        if len(values) >= 3:
            _, p = shapiro(values)
            if p < 0.05:
                normal = False
                break

    _, p_levene = levene(*group_values)
    equal_var = p_levene >= 0.05
    return 'anova' if normal and equal_var else 'kruskal'

# === Effektgrößenberechnung ===
def eta_squared_anova(groups, f_stat):
    k = len(groups)
    n = sum(len(g) for g in groups)
    df_between = k - 1
    df_within = n - k
    return (f_stat * df_between) / (f_stat * df_between + df_within)

def epsilon_squared_kruskal(groups, h_stat):
    n = sum(len(g) for g in groups)
    return (h_stat - len(groups) + 1) / (n - 1)

def rank_biserial_effect(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)

# === Analysefunktion ===
def analyze_folder(folder_path):
    csv_in = os.path.join(folder_path, "persona_results.csv")
    if not os.path.isfile(csv_in):
        print(f"Kein persona_results.csv in {folder_path}")
        return

    df = pd.read_csv(csv_in)
    stats_csv = os.path.join(folder_path, "statistical_results.csv")
    sig_csv = os.path.join(folder_path, "significant_results.csv")
    tukey_csv = os.path.join(folder_path, "tukey_results.csv")
    tukey_sig_csv = os.path.join(folder_path, "tukey_significant_results.csv")

    stat_results, tukey_results_all = [], []

    for group_col in GROUP_COLUMNS:
        for metric in METRICS:
            method = check_distribution_and_select_test(df, group_col, metric)
            grouped_data = [g[metric].values for _, g in df.groupby(group_col)]

            if method == 'anova':
                stat, p_val = f_oneway(*grouped_data)
                effect_size = eta_squared_anova(grouped_data, stat)
            elif method == 'kruskal':
                stat, p_val = kruskal(*grouped_data)
                effect_size = epsilon_squared_kruskal(grouped_data, stat)
            elif method == 'mannwhitney':
                groups = list(df[group_col].unique())
                x = df[df[group_col] == groups[0]][metric]
                y = df[df[group_col] == groups[1]][metric]
                stat, p_val = mannwhitneyu(x, y, alternative='two-sided')
                effect_size = rank_biserial_effect(stat, len(x), len(y))
            else:
                continue

            stat_results.append({
                'group': group_col,
                'metric': metric,
                'method': method,
                'stat': stat,
                'p': p_val,
                'effect_size': effect_size
            })

            if p_val < 0.05:
                if method == 'anova':
                    tukey = pairwise_tukeyhsd(df[metric], df[group_col])
                    tmp = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                    tmp['metric'] = metric
                    tmp['grouping'] = group_col
                    tukey_results_all.append(tmp)
                elif method == 'kruskal':
                    dunn = sp.posthoc_dunn(df, val_col=metric, group_col=group_col, p_adjust='bonferroni')
                    dunn = dunn.reset_index().melt(id_vars='index')
                    dunn.columns = ['group1', 'group2', 'p-adj']
                    dunn = dunn[dunn['group1'] != dunn['group2']]
                    dunn['reject'] = dunn['p-adj'] < 0.05
                    dunn['metric'] = metric
                    dunn['grouping'] = group_col
                    dunn['meandiff'] = dunn.apply(lambda row: df[df[group_col]==row['group1']][metric].mean() - df[df[group_col]==row['group2']][metric].mean(), axis=1)
                    tukey_results_all.append(dunn)

    df_stat = pd.DataFrame(stat_results)
    df_stat.to_csv(stats_csv, index=False)
    df_stat[df_stat.p < 0.05].to_csv(sig_csv, index=False)

    if tukey_results_all:
        df_tuk = pd.concat(tukey_results_all, ignore_index=True)
        df_tuk.to_csv(tukey_csv, index=False)
        df_tuk[df_tuk['reject'] == True] \
            .sort_values(['metric', 'grouping', 'meandiff'], ascending=[True, True, False]) \
            .to_csv(tukey_sig_csv, index=False)

    print(f"Analyse abgeschlossen für: {folder_path}")

if __name__ == "__main__":
    for sub in os.listdir(ROOT_RESULTS):
        folder = os.path.join(ROOT_RESULTS, sub)
        if os.path.isdir(folder):
            print(f"Verarbeite {folder}")
            analyze_folder(folder)