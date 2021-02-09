__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/02/09 11:30:00"

import sys

from matplotlib import pyplot as plt

## Expected running with one parameter with is the path to the file
## with generated report from validation training phase. Parse that file and plot it.

file_name = sys.argv[1]

epochs = []
loss_list = []
valid_list = []
with open(file_name, "r") as file:
    for l in file:
        # Epoch: 41699, loss: 312.96, validation loss: 330.58
        line = l.split()
        if len(line) == 0:
            continue
        if line[0] == "Epoch:":
            epochs.append(int(line[1].split(',')[0]))
            loss_list.append(float(line[3].split(',')[0]))
            valid_list.append(float(line[6]))
print(valid_list[-3:])
print(loss_list[-3:])
print(epochs[-3:])
location = "validation_plot.png"
if len(sys.argv) > 2:
    location = sys.argv[2]
plt.plot(epochs, loss_list)
plt.plot(epochs, valid_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

print("generating training validation plot into", location)
plt.savefig(location)