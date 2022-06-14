import matplotlib.pyplot as plt
import json
plt.rcParams.update({'figure.max_open_warning': 0})
with open('summary.json', 'r') as fin:
    summary = json.load(fin)
#fig, ax1 = plt.subplots()
for res_path, sum_d in summary.items():
    plt.clf()
    
    sizes = []
    losses = []
    accs = []
    avg_cers = []
    for size in sorted(sum_d.keys(), key=lambda x: int(x)):
        d = sum_d[size]
        sizes.append(int(size))
        losses.append(d['loss'])
        accs.append(d['acc'])
        avg_cers.append(d['avg_cer'])

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(sizes, accs, 'g-')
    ax2.plot(sizes, avg_cers, 'b-')

    ax1.set_xlabel(f'batch has run')
    ax1.set_ylabel('accuracy', color='g')
    ax2.set_ylabel('average CER', color='b')
    plt.grid(True)
    #plt.title(res_path)
    fig_prefix = res_path.replace('/', '_').replace('\\', '_')[:-4]
    print("save ", f"{fig_prefix}_fig.png")
    plt.savefig(f"{fig_prefix}_fig.png")
    #plt.show()
