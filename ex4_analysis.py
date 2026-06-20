import warnings
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from sklearn.decomposition import PCA

DATA_PATH = 'ps4_data.mat'
N_COMP    = 50
RNG_SEED  = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path):
    # MATLAB v7.3 .mat files are HDF5. h5py reads dimensions reversed
    # (MATLAB Fortran order → C order), so we transpose arrays back.
    with h5py.File(path, 'r') as f:
        data = {
            'N':               int(f['N'][0, 0]),
            'g':               float(f['g'][0, 0]),
            't':               f['t'][:].squeeze(),
            'target_t':        f['target_t'][:].squeeze(),
            'R':               f['R'][:].T,
            'R0':              f['R0'][:].T,
            'target_activity': f['target_activity'][:].T,
            'J0':              f['J0'][:].T,
            'J':               f['J'][:].T,
        }
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def heatmap(ax, data, title, t_vec):
    vabs = np.percentile(np.abs(data), 98)
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                   extent=[t_vec[0], t_vec[-1], data.shape[0], 0],
                   vmin=-vabs, vmax=vabs)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Unit')
    plt.colorbar(im, ax=ax, fraction=0.02)


def fit_pca(X, n):
    """X: N x T. Returns (pca, projection n x T, cumulative explained variance)."""
    pca = PCA(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        pca.fit(X.T)
        proj = pca.transform(X.T).T
    cevr = pca.explained_variance_ratio_.cumsum()
    return pca, proj, cevr


def mark_endpoints(ax, x, y, z=None, s=60):
    if z is None:
        ax.scatter(x[0],  y[0],  marker='o', s=s, color='green', zorder=5, label='Start')
        ax.scatter(x[-1], y[-1], marker='s', s=s, color='red',   zorder=5, label='End')
    else:
        ax.scatter(x[0],  y[0],  z[0],  marker='o', s=s, color='green')
        ax.scatter(x[-1], y[-1], z[-1], marker='s', s=s, color='red')


def savefig(name):
    plt.savefig(name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {name}')


# ---------------------------------------------------------------------------
# Section 3.1 — Activity
# ---------------------------------------------------------------------------

def section_3_1(R, R0, target_activity, t, target_t, unit_idx):
    T_rnn    = R.shape[1]
    T_target = target_activity.shape[1]
    step     = T_rnn // T_target

    # 3.1.1 — trained RNN vs target
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, u in zip(axes, unit_idx):
        ax.plot(t, R[u], color='tab:blue', lw=1.2, label=f'Trained RNN (unit {u})')
        ax.plot(target_t, target_activity[u], color='tab:orange', lw=1.2, ls='--',
                label='Target')
        ax.set_ylabel('Activity')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Time')
    fig.suptitle('3.1.1 — Trained RNN vs Target (4 units)', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_1_1_trained_vs_target.png')

    # 3.1.2 — trained vs untrained RNN
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, u in zip(axes, unit_idx):
        ax.plot(t, R0[u], color='tab:gray', lw=1.0, alpha=0.55, label=f'Untrained (unit {u})')
        ax.plot(t, R[u],  color='tab:blue', lw=1.2, alpha=0.90, label=f'Trained  (unit {u})')
        ax.set_ylabel('Activity')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Time')
    fig.suptitle('3.1.2 — Trained vs Untrained RNN (4 units)', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_1_2_trained_vs_untrained.png')

    # 3.1.3 — full network heatmaps (down-sampled to target time grid)
    R_ds  = R[:,  ::step][:, :T_target]
    R0_ds = R0[:, ::step][:, :T_target]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    heatmap(axes[0], R_ds,            'Trained RNN (down-sampled)',   target_t)
    heatmap(axes[1], R0_ds,           'Untrained RNN (down-sampled)', target_t)
    heatmap(axes[2], target_activity, 'Target activity',              target_t)
    fig.suptitle('3.1.3 — Full network activity heatmaps', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_1_3_heatmaps.png')

    return step


# ---------------------------------------------------------------------------
# Section 3.2 — Dynamics (PCA)
# ---------------------------------------------------------------------------

def section_3_2(R, R0, target_activity, t, target_t, step):
    T_target = target_activity.shape[1]

    pca_R,   R_proj,   cevr_R   = fit_pca(R,               N_COMP)
    pca_R0,  R0_proj,  cevr_R0  = fit_pca(R0,              N_COMP)
    pca_tgt, tgt_proj, cevr_tgt = fit_pca(target_activity, N_COMP)

    pcs_R   = np.searchsorted(cevr_R,   0.90) + 1
    pcs_R0  = np.searchsorted(cevr_R0,  0.90) + 1
    pcs_tgt = np.searchsorted(cevr_tgt, 0.90) + 1
    print(f'Trained RNN   — PCs for 90% var: {pcs_R}')
    print(f'Untrained RNN — PCs for 90% var: {pcs_R0}')
    print(f'Target        — PCs for 90% var: {pcs_tgt}')

    datasets = [
        (R_proj,   'Trained RNN',   'tab:blue'),
        (R0_proj,  'Untrained RNN', 'tab:gray'),
        (tgt_proj, 'Target',        'tab:orange'),
    ]

    # 3.2.1a — 2D trajectories (own PC space)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (proj, title, color) in zip(axes, datasets):
        ax.plot(proj[0], proj[1], color=color, lw=0.8)
        mark_endpoints(ax, proj[0], proj[1])
        ax.set_title(title)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend(fontsize=8)
    fig.suptitle('3.2.1 — 2D PC trajectories (own space)', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_2_1a_2d_trajectories.png')

    # 3.2.1b — 3D trajectories (own PC space)
    fig = plt.figure(figsize=(15, 4))
    for i, (proj, title, color) in enumerate(datasets, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.plot(proj[0], proj[1], proj[2], color=color, lw=0.8)
        mark_endpoints(ax, proj[0], proj[1], proj[2])
        ax.set_title(title)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
    fig.suptitle('3.2.1 — 3D PC trajectories (own space)', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_2_1b_3d_trajectories.png')

    # 3.2.2 — all three in trained RNN's PC space
    R_ds_full  = R[:,  ::step][:, :T_target]
    R0_ds_full = R0[:, ::step][:, :T_target]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        R_in_R   = pca_R.transform(R_ds_full.T).T
        R0_in_R  = pca_R.transform(R0_ds_full.T).T
        tgt_in_R = pca_R.transform(target_activity.T).T

    shared = [
        (R_in_R,   'Trained RNN',   'tab:blue'),
        (R0_in_R,  'Untrained RNN', 'tab:gray'),
        (tgt_in_R, 'Target',        'tab:orange'),
    ]

    fig = plt.figure(figsize=(14, 5))
    ax2 = fig.add_subplot(1, 2, 1)
    for proj, label, color in shared:
        ax2.plot(proj[0], proj[1], color=color, lw=0.9, label=label)
        ax2.scatter(proj[0, 0],  proj[1, 0],  marker='o', s=40, color=color)
        ax2.scatter(proj[0, -1], proj[1, -1], marker='s', s=40, color=color)
    ax2.set_xlabel('PC 1 (Trained RNN space)')
    ax2.set_ylabel('PC 2 (Trained RNN space)')
    ax2.set_title('2D — shared PC space')
    ax2.legend()

    ax3 = fig.add_subplot(1, 2, 2, projection='3d')
    for proj, label, color in shared:
        ax3.plot(proj[0], proj[1], proj[2], color=color, lw=0.9, label=label)
    ax3.set_xlabel('PC 1')
    ax3.set_ylabel('PC 2')
    ax3.set_zlabel('PC 3')
    ax3.set_title('3D — shared PC space')
    ax3.legend(fontsize=8)

    fig.suptitle('3.2.2 — All trajectories in trained RNN PC space', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_2_2_shared_pc_space.png')

    # 3.2.3 — cumulative explained variance
    fig, ax = plt.subplots(figsize=(8, 5))
    pcs = np.arange(1, N_COMP + 1)
    ax.plot(pcs, cevr_R,   label='Trained RNN',   color='tab:blue')
    ax.plot(pcs, cevr_R0,  label='Untrained RNN', color='tab:gray')
    ax.plot(pcs, cevr_tgt, label='Target',        color='tab:orange')
    ax.axhline(0.90, color='black', ls='--', lw=0.8, label='90% threshold')
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative explained variance ratio')
    ax.set_title('3.2.3 — Cumulative explained variance')
    ax.legend()
    plt.tight_layout()
    savefig('fig_3_2_3_cumvar.png')


# ---------------------------------------------------------------------------
# Section 3.3 — Connectivity
# ---------------------------------------------------------------------------

def section_3_3(J, J0, N, g):
    # 3.3.1 — norm of weight change vs random baseline
    delta_J_norm = np.linalg.norm(J - J0)
    rng2  = np.random.default_rng(0)
    sigma = g / np.sqrt(N)
    random_norms = [
        np.linalg.norm(rng2.normal(0, sigma, (N, N)) - rng2.normal(0, sigma, (N, N)))
        for _ in range(50)
    ]
    rand_mean = np.mean(random_norms)
    rand_std  = np.std(random_norms)
    print(f'||J - J0||                   = {delta_J_norm:.2f}')
    print(f'||rand1 - rand2|| mean ± std = {rand_mean:.2f} ± {rand_std:.2f}')
    print(f'Ratio (actual / random mean) = {delta_J_norm / rand_mean:.3f}')

    # 3.3.2 — weight histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(J0.ravel(), bins=200, color='tab:gray', edgecolor='none')
    axes[0].set_yscale('log')
    axes[0].set_title('Untrained weights J0')
    axes[0].set_xlabel('Weight value')
    axes[0].set_ylabel('Count (log scale)')

    axes[1].hist(J.ravel(), bins=200, color='tab:blue', edgecolor='none')
    axes[1].set_yscale('log')
    axes[1].set_title('Trained weights J')
    axes[1].set_xlabel('Weight value')
    axes[1].set_ylabel('Count (log scale)')

    fig.suptitle('3.3.2 — Weight distribution (log-scale)', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_3_2_weight_histograms.png')

    # 3.3.3 — connectivity heatmaps
    vmax = np.percentile(np.abs(np.concatenate([J.ravel(), J0.ravel()])), 99)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in zip(axes, [J0, J], ['Untrained J0', 'Trained J']):
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Source unit (column = outgoing)')
        ax.set_ylabel('Target unit')
        plt.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle('3.3.3 — Connectivity matrices', fontsize=13)
    plt.tight_layout()
    savefig('fig_3_3_3_connectivity_heatmaps.png')

    # 3.3.4 — trained columns (column mean-square > 0.5)
    col_ms       = (J ** 2).mean(axis=0)
    trained_mask = col_ms > 0.5
    n_trained    = trained_mask.sum()
    print(f'Trained units (col MS > 0.5): {n_trained} / {N}')

    J_trained  = J[:, trained_mask]
    sort_order = np.argsort(np.argmax(np.abs(J_trained), axis=0))
    J_sorted   = J_trained[:, sort_order]

    vmax2 = np.percentile(np.abs(J_sorted), 99)
    fig, ax = plt.subplots(figsize=(max(6, n_trained // 8), 6))
    im = ax.imshow(J_sorted, aspect='auto', cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
    ax.set_title(f'3.3.4 — Trained columns only ({n_trained} units, col MS > 0.5)')
    ax.set_xlabel('Trained source unit (sorted by peak row)')
    ax.set_ylabel('Target unit')
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    savefig('fig_3_3_4_trained_columns.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading data...')
    d = load_data(DATA_PATH)
    N               = d['N']
    g               = d['g']
    t               = d['t']
    target_t        = d['target_t']
    R               = d['R']
    R0              = d['R0']
    target_activity = d['target_activity']
    J0              = d['J0']
    J               = d['J']
    print(f'N={N}, g={g:.2f}  |  R: {R.shape}  |  target: {target_activity.shape}')

    rng      = np.random.default_rng(RNG_SEED)
    unit_idx = rng.choice(N, size=4, replace=False)
    print(f'Selected units: {unit_idx}')

    print('\n--- Section 3.1: Activity ---')
    step = section_3_1(R, R0, target_activity, t, target_t, unit_idx)

    print('\n--- Section 3.2: Dynamics ---')
    section_3_2(R, R0, target_activity, t, target_t, step)

    print('\n--- Section 3.3: Connectivity ---')
    section_3_3(J, J0, N, g)

    print('\nDone.')


if __name__ == '__main__':
    main()
