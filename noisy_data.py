import torch
import os


print(os.getcwd())
all_labels = torch.load("./data/CIFAR-10_human.pt")

clean_label = all_labels["clean_label"]
worst_label = all_labels["worse_label"]
aggregate_label = all_labels["aggre_label"]
random_label1 = all_labels["random_label1"]
random_label2 = all_labels["random_label2"]
random_label3 = all_labels["random_label3"]

print(clean_label[:10])
print(worst_label[:10])