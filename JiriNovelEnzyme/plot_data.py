"""
For Jiri Damborsky preparation of plot graph for NovelEnzyme conference
date: 01/02/2023
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd

from adjustText import adjust_text


def get_indexes_of_accessions(accessions_list, accessions):
    """ Get just indexes to embeddings mus values """
    # indexes = [accessions_list.index(acc) for acc in accessions]
    indexes = []
    for acc in accessions:
        try:
            indexes.append(accessions_list.index(acc))
        except:
            pass
    return indexes


df = pd.read_excel("hlds_mapping.xlsx", sheet_name='Sheet1')

# Green points
df_greens = df.loc[df['Characterized?'] == 'yes']
green_acc_to_name = {row['Accession No.']: row['Name'] for _, row in df_greens.iterrows()}
green_name_to_acc = {row['Name']: row['Accession No.'] for _, row in df_greens.iterrows()}
green_vals = list(green_name_to_acc.keys())
# Red points
red_vals = ["DrxA"]
df_red = df.loc[df['Name'].isin(red_vals)]
red_acc_to_name = {row['Accession No.']: row['Name'] for _, row in df_red.iterrows()}
red_name_to_acc = {row['Name']: row['Accession No.'] for _, row in df_red.iterrows()}
# Yellow points
yellow_vals = ['DspB', 'DskA', 'DmuA', 'DpxA', 'DgarA', 'DhliA', 'DhsA', 'DhmuA', 'DdsA', 'DmoxA', 'DdpA', 'DphexA']
df_yellow = df.loc[df['Name'].isin(yellow_vals)]
yel_acc_to_name = {row['Accession No.']: row['Name'] for _, row in df_yellow.iterrows()}
yel_name_to_acc = {row['Name']: row['Accession No.'] for _, row in df_yellow.iterrows()}
assert len(yellow_vals) == len(list(yel_name_to_acc.values()))
# Blue points
blue_vals = ['DcagA', 'DpproA', 'DshA', 'DnlA']
df_blue = df.loc[df['Name'].isin(blue_vals)]
blue_acc_to_name = {row['Accession No.']: row['Name'] for _, row in df_blue.iterrows()}
blue_name_to_acc = {row['Name']: row['Accession No.'] for _, row in df_blue.iterrows()}
assert len(blue_vals) == len(list(blue_name_to_acc.values()))
# Black points
black_vals = ['DphxA', 'DbbA', 'DtacA', 'DmgoA']
df_black = df.loc[df['Name'].isin(black_vals)]
black_acc_to_name = {row['Accession No.']: row['Name'] for _, row in df_black.iterrows()}
black_name_to_acc = {row['Name']: row['Accession No.'] for _, row in df_black.iterrows()}
assert len(black_vals) == len(list(black_name_to_acc.values()))

# Read Hon subfamilies
dfhon = pd.read_excel("Hon_2021_HTS2_selection_dataset_from_2017.xlsx", sheet_name="FULL DATASET")
accessions1 = dfhon.loc[dfhon["Subfamily"] == 1]["Accession"].tolist()
accessions2 = dfhon.loc[dfhon["Subfamily"] == 2]["Accession"].tolist()
accessions3 = dfhon.loc[dfhon["Subfamily"] == 3]["Accession"].tolist()
accessions4 = dfhon.loc[dfhon["Subfamily"] == 4]["Accession"].tolist()

with open("embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

# Subfamilies coordinates
# coordinates_sub1 = get_indexes_of_accessions(accessions_list=embeddings['keys'], accessions=accessions1)
# coordinates_sub2 = get_indexes_of_accessions(accessions_list=embeddings['keys'], accessions=accessions2)
# coordinates_sub3 = get_indexes_of_accessions(accessions_list=embeddings['keys'], accessions=accessions3)
# coordinates_sub3 = get_indexes_of_accessions(accessions_list=embeddings['keys'], accessions=accessions3)

coords_list = []
for acc_sub in [accessions1, accessions2, accessions3, accessions4]:
    coords_list.append(get_indexes_of_accessions(accessions_list=embeddings['keys'], accessions=acc_sub))

green_idx = get_indexes_of_accessions(embeddings['keys'], list(green_acc_to_name.keys()))
red_idx = get_indexes_of_accessions(embeddings['keys'], list(red_acc_to_name.keys()))
blue_idx = get_indexes_of_accessions(embeddings['keys'], list(blue_acc_to_name.keys()))
black_idx = get_indexes_of_accessions(embeddings['keys'], list(black_acc_to_name.keys()))

# Plot plus annotation of selected points
plt.ylim([-6, 9])
plt.xlim([-7, 4])
plt.axis('off')
plt.scatter(embeddings['mu'][:, 0], embeddings['mu'][:, 1], marker='o', edgecolors='black', color='grey')
ts = []
for idx, c, labels in [(green_idx, 'g', green_vals), (red_idx, 'r', red_vals),
                     (blue_idx, 'blue', blue_vals), (black_idx, 'black', black_vals)]:
    plt.scatter(embeddings['mu'][idx, 0], embeddings['mu'][idx, 1], marker='o', edgecolors='black', color=c)
    for i, index in enumerate(idx):
        ts.append(plt.text(embeddings['mu'][index, 0], embeddings['mu'][index, 1], labels[i]))

adjust_text(ts, x=embeddings['mu'][:, 0], y=embeddings['mu'][:, 1],
            force_points=0.2, arrowprops=dict(arrowstyle='-', color='black'))
plt.show()

# Plot subfamilies
# plt.scatter(embeddings['mu'][coords_list[0], 0], embeddings['mu'][coords_list[0], 1], marker='o', edgecolors='black', color='#F1F2E1')
# plt.scatter(embeddings['mu'][coords_list[1], 0], embeddings['mu'][coords_list[1], 1], marker='o', edgecolors='black', color='#EDEBFE')
# plt.scatter(embeddings['mu'][coords_list[2], 0], embeddings['mu'][coords_list[2], 1], marker='o', edgecolors='black', color='#EEFBFA')
# plt.scatter(embeddings['mu'][coords_list[3], 0], embeddings['mu'][coords_list[3], 1], marker='o', edgecolors='black', color='#FFF4F7')
plt.scatter(embeddings['mu'][coords_list[0], 0], embeddings['mu'][coords_list[0], 1], marker='o', edgecolors='black', color='red')
plt.scatter(embeddings['mu'][coords_list[1], 0], embeddings['mu'][coords_list[1], 1], marker='o', edgecolors='black', color='blue')
plt.scatter(embeddings['mu'][coords_list[2], 0], embeddings['mu'][coords_list[2], 1], marker='o', edgecolors='black', color='green')
plt.scatter(embeddings['mu'][coords_list[3], 0], embeddings['mu'][coords_list[3], 1], marker='o', edgecolors='black', color='yellow')
plt.show()
