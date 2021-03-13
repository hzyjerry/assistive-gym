import matplotlib.pyplot as plt
import os

metrics = ["reward", "mean force", "success"]
keywords = ["Reward total: ", "mean force: ", "task success: "]

def read_file(filename):
    profs = {key: [] for key in metrics}
    with open(filename) as f:
        for line in f:
            if keywords[0] in line:
                line = line.strip()
                for m, kw in zip(metrics, keywords):
                    line = line.split(kw)[1]
                    val = float(line.split(",", 1)[0])
                    profs[m].append(val)
                    if len(line.split(",", 1)) > 1:
                        line = line.split(",", 1)[1]
    return profs

def plot_files(folder):
    all_profs = {}
    num_files = 0
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            profs = read_file(os.path.join(folder, file))
            all_profs[file] = profs
            num_files += 1
    # print(all_profs)
    for m in metrics:
        fig, ax = plt.subplots(figsize=(3 * num_files, 4))
        all_vals = []
        all_keys = []
        # for key in all_profs.keys():
        for key in ["human-v0217_h1-robot-scratch.txt", "human-v0217_h1-robot-v0217_0.txt", "human-v0217_h1-robot-personalized.txt"]:
            profs = all_profs[key]
            all_keys.append(key.replace("-robot", "\nrobot").replace(".txt", ""))
            all_vals.append(profs[m])
        ax.boxplot(all_vals, whis=[5, 95], meanline=True, showmeans=True)
        ax.set_xticklabels(all_keys, fontsize=8)
        plt.savefig(f"{folder}/{m.replace(' ', '_')}.png")



if __name__ == "__main__":
    folder = "data/210226"
    plot_files(folder)