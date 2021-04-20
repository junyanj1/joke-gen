import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir= 'g100d1k10'
filename = 'logs_gan_g100d1k10.txt'
epochs = 100
file1 = open(os.path.join('SeqGAN/'+dir, filename), 'r')
lines = file1.readlines()

accs = []
loss = []

ep_accs = []
ep_loss = []

for line in lines:
    if "val_acc" in line:
        acc = line.strip().split(" ")[14]
        loss_line = line.strip().split(" ")[8][:-1]
        ep_accs.append(float(acc))
        ep_loss.append(float(loss_line))
    elif "EPOCH" in line:
        if len(ep_accs) > 0:
            accs.append(ep_accs)
        if len(ep_loss) > 0:
            loss.append(ep_loss)
        ep_accs = []
        ep_loss = []

file1.close()

accs_flat = np.array(accs).flatten()
loss_flat = np.array(loss).flatten()

x = np.linspace(epochs,(epochs+len(accs[-1])/epochs)+50,len(accs_flat))

df = pd.read_csv(os.path.join('SeqGAN/' + dir, 'nll_per_epoch.csv'))

plt.figure(figsize=(8,5))

plt.plot(df['epoch'], df['loss'], label="Generator")
plt.ylim(1,3.9)
plt.legend()
plt.vlines(100, 0,10, linestyles="dashed", colors="black")
plt.xlabel("Epochs")
plt.ylabel("NLL")
plt.title("(b) g-steps=1, d-steps=1, $k$=10")
plt.savefig("{}_loss.png".format(dir))
plt.show()

plt.plot(x, loss_flat)
plt.ylim(0.5,1.1)
plt.xlabel("Epochs")
plt.plot(x, accs_flat, label="Discriminator", color="orange")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("{}_acc.png".format(dir))
plt.show()