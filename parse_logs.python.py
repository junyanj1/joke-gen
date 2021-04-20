import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'logs_gan_d1_k10.txt'
epochs = 20

# with open(os.path.join('SeqGAN', filename), mode='r') as fin:
file1 = open(os.path.join('SeqGAN', filename), 'r')
lines = file1.readlines()

accs = []

ep_accs = []
for line in lines:
    if "val_acc" in line:
        acc = line.strip().split(" ")[14]
        ep_accs.append(float(acc))
    elif "EPOCH" in line:
        if len(ep_accs) > 0:
            accs.append(ep_accs)
        ep_accs = []
print(accs)

accs_flat = np.array(accs).flatten()
print(len(accs_flat))

x = np.linspace(0,epochs+len(accs[-1])/epochs,len(accs_flat))


df = pd.read_csv(os.path.join('SeqGAN', 'nll_per_epoch.csv'))
print(df)
plt.plot(df['epoch'], df['loss'])

plt.ylim(0.4,4)
plt.plot(x, accs_flat)
plt.show()

file1.close()