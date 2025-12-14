import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
})


def moving_average(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode="valid")

def extract_metric(arr, key):
    return np.array([ep[key] for ep in arr], dtype=float)

def load_train_rewards(path):
    data = np.load(path, allow_pickle=True)

    if np.issubdtype(data.dtype, np.number):
        return np.asarray(data, dtype=float)

    if isinstance(data[0], dict):
        return np.array([ep["reward"] for ep in data], dtype=float)

    raise ValueError(f"Unrecognized format in {path}")

# =========================
# Configuration
# =========================
suffixes = [
    "baseline",
    "deeptamer_keyboard",
    "deeptamer_voice",
    "deeptamer_vad",
]

labels = {
    "baseline": "DDPG",
    "deeptamer_keyboard": "DTK",
    "deeptamer_voice": "DTW",
    "deeptamer_vad": "DTP",
}

results_folder = 'results/train_rewards/'
all_metrics_folder = 'results/all_metrics/'

reward_files = {
    "baseline": "train_rewards_ddpg.npy",
    "deeptamer_keyboard": "train_rewards_dtk.npy",
    "deeptamer_voice": "train_rewards_dtw.npy",
    "deeptamer_vad": "train_rewards_dtp.npy",
}

metrics_files = {
    "baseline": "all_metrics_ddpg.npy",
    "deeptamer_keyboard": "all_metrics_dtk.npy",
    "deeptamer_voice": "all_metrics_dtw.npy",
    "deeptamer_vad": "all_metrics_dtp.npy",
}

ma_window = 3

#Training rewards
plt.figure(figsize=(8, 4))

for suffix in suffixes:
    rewards = load_train_rewards(results_folder+reward_files[suffix])
    plt.plot(
        moving_average(rewards, ma_window),
        label=labels[suffix],
    )

plt.xlabel("Training Episode")
plt.ylabel("Cumulative Reward")
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/reward_vs_episode_ma.eps", format="eps")
plt.show()

# exploration
data_box = []
box_labels = []

for suffix in suffixes:
    metrics = np.load(all_metrics_folder+metrics_files[suffix], allow_pickle=True)
    unique_regions = extract_metric(metrics, "unique_regions")
    data_box.append(unique_regions)
    box_labels.append(labels[suffix])

plt.figure(figsize=(6, 4))
plt.boxplot(
    data_box,
    labels=box_labels,
    widths=0.5,
    showfliers=True,
    patch_artist=False,
)

plt.ylabel("Unique regions visited per episode")
plt.ylim(0, max(d.max() for d in data_box) + 0.5)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/unique_regions_boxplot.eps", format="eps")
plt.show()
