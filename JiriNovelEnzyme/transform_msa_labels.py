"""
Transformation of MSa Labels to the accessions
"""

mapping = {"acc": [], "lab": []}
with open("identified_targets_headers.txt") as file:
    for line in file.readlines():
        splitted = line.split()
        accession, label = splitted[0], " ".join(splitted[1:])
        mapping["acc"].append(accession)
        mapping["lab"].append(label)


# translate to accession and write to the file simultaneously
def translate_label_to_accession(label):
    """ Find correct accession for this label """
    label = label.rstrip()
    try:
        index = mapping["lab"].index(label)
        return mapping["acc"][index]
    except ValueError:
        # label can be truncated
        for i, lab_full in enumerate(mapping["lab"]):
            if label in lab_full:
                return mapping["acc"][i]
        print(f"Label {label} not found")


with open("hon_paper_dataset.fas") as file:
    with open("hon_accession_dataset.fa", "w") as hon:
        current_acc = None
        current_seq = ""
        for line in file.readlines():
            if ">" in line:
                if current_acc is not None:
                    hon.write(f">{current_acc}\n{current_seq}")
                    current_seq = ""
                current_acc = translate_label_to_accession(line)
                continue
            current_seq += line
        # write last sequence
        hon.write(f">{current_acc}\n{current_seq}")
