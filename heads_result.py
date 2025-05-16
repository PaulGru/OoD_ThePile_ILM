import os
import numpy as np
import re
import matplotlib.pyplot as plt

heads_dir = "lm_heads"
files = sorted([f for f in os.listdir(heads_dir) if f.endswith(".npy")])

# Extract all unique steps using regex to pull the trailing number before ".npy"
steps = sorted(list(set(
    int(re.search(r'(\d+)\.npy$', f).group(1))
    for f in files if re.search(r'(\d+)\.npy$', f)
)))
# Extract the environment as everything except the "-<step>" part
envs = sorted(list(set(
    f.rsplit("-", 1)[0] for f in files
)))

all_heads = {step: {} for step in steps}

for f in files:
    # Split off the last segment using rsplit so that extra dashes in the environment don’t matter.
    env = f.rsplit("-", 1)[0]
    # Use regex to extract the numeric step
    match = re.search(r'(\d+)\.npy$', f)
    if match:
        step = int(match.group(1))
    else:
        print(f"Filename {f} does not contain a valid numeric step. Skipping.")
        continue
    vec = np.load(os.path.join(heads_dir, f))
    all_heads[step][env] = vec.flatten()

# Calculate the mean pairwise distances between head vectors at each step
mean_distances = []
for step in steps:
    heads = all_heads[step]
    env_vectors = list(heads.values())
    n = len(env_vectors)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(env_vectors[i] - env_vectors[j])
            distances.append(dist)
    mean_distances.append(np.mean(distances))

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(steps, mean_distances, marker='o')
plt.xlabel("Step")
plt.ylabel("Mean Pairwise Distance Between LM Heads")
plt.title("Divergence of LM Heads Over Training Steps")
plt.grid(True)
plt.tight_layout()
plt.savefig("lm_heads_divergence.png", dpi=300)
