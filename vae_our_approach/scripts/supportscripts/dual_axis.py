import csv
import argparse
import matplotlib
import matplotlib.pyplot as plt


def parse_csv(file):
    probs = []
    wt_identity = []
    closest_identity = []
    residues90 = []
    indels_cnt = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            probs.append(float(row['Probability of observation'])*100)
            wt_identity.append(float(row['Query identity [%]']))
            closest_identity.append(float(row['Closest identity [%]']))
            residues90.append(int(row['Residues count of probabilities above 0.9']))
            indels_cnt.append(int(row['Count of indels']))
    return probs, wt_identity, closest_identity, residues90, indels_cnt


def plot_dual_axis(prs, wt, cl, res, inds, file):
    plt.figure(figsize=(30, 6), dpi=400)
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rc('xtick', labelsize=8)

    ancs = range(len(prs))
    fig, ax = plt.subplots()
    fig.set_size_inches(28.5, 15.5)
    plt.xticks(ancs, ancs)
    plt.xticks(ancs, [x if i != 30 else 'Q' for i, x in enumerate(ancs)])
    ax.plot(ancs, prs, color="red", marker="o", label='Probability')
    ax.plot(ancs, wt, color="green", marker="v", label='WT')
    ax.plot(ancs, cl, color="orange", marker="^", label='Closest')
    ax.set_xlabel("successors/ancestors number", fontsize=14)
    ax.set_ylabel("Percentage [%]", color="red", fontsize=20)
    ax.legend(loc='upper left', bbox_to_anchor=(0.15, 1.07),
              ncol=3, fancybox=True, shadow=True, fontsize=18)

    # Second y axis
    ax2 = ax.twinx()
    ax2.set_ylim([0,314])
    ax2.plot(ancs, res, color="blue", marker="s", label='Cnt of residues above 90% likelihood')
    ax2.plot(ancs, inds, color="black", marker="x", label='Indel cnt')
    ax2.set_ylabel("Residue cnt", color="blue", fontsize=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.8, 1.07),
               ncol=2, fancybox=True, shadow=True, fontsize=18)
    plt.vlines(30, 0, 314, color='black', linestyles='dashed')
    # save the plot as a file
    name = (file.split('/')[-1]).split('.')[0] + '.jpg'
    print(" Dual axis report : saving plot into", name)
    fig.savefig(name,
                format='jpeg',
                dpi=400,
                bbox_inches='tight')


# Program parse ready
parser = argparse.ArgumentParser(description='Script for preparing plot with dual axis for mutants selection')
parser.add_argument("--csv", help="Csv file with data")

# Run script
args = parser.parse_args()
pr, wt, cl, res, inds = parse_csv(args.csv)
plot_dual_axis(pr, wt, cl, res, inds, args.csv)
