"""Training curves and qualitative prediction panels."""
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

CURVES = [("loss", "Loss"), ("f1_score_metric", "F1-Score"), ("soft_dice_score", "Dice"),
          ("iou_score", "IoU"), ("precision_metric", "Precision"), ("recall_metric", "Recall")]


def plot_history(history, path, title="Train vs Val"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=14)
    for ax, (key, label) in zip(axes.ravel(), CURVES):
        if key in history:
            ax.plot(history[key], label="Train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="Val")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prediction_panel(model, dataset, path, threshold=0.5, n_samples=3, title=""):
    """RGB | (DEM) | Ground truth | Prediction for small / median / large landslides.

    Taken from the first batch of ``dataset`` (so use an un-shuffled val/test set).
    """
    images, masks = next(iter(dataset.take(1)))
    images, masks = images.numpy(), masks.numpy()
    preds = model.predict(images, verbose=0).squeeze(-1)

    sums = masks.sum(axis=(1, 2, 3))
    order = np.argsort(sums)
    nz = [i for i in order if sums[i] > 0]
    if len(nz) >= n_samples:
        sel = [nz[0], nz[len(nz) // 2], nz[-1]][:n_samples]
    else:
        sel = (nz + [i for i in order if i not in nz])[:n_samples]
    sel = sel[:len(images)]

    has_dem = images.shape[-1] == 4
    cols = ["RGB"] + (["DEM"] if has_dem else []) + ["Ground truth", "Prediction"]
    fig, axes = plt.subplots(len(sel), len(cols), figsize=(3 * len(cols), 3 * len(sel)),
                             squeeze=False)
    px = images.shape[1] * images.shape[2]
    for r, idx in enumerate(sel):
        c = 0
        axes[r, c].imshow(images[idx][..., :3])
        axes[r, c].text(4, 18, f"LS: {100 * sums[idx] / px:.1f}%", color="white",
                        backgroundcolor="crimson", fontsize=7, fontweight="bold")
        c += 1
        if has_dem:
            axes[r, c].imshow(images[idx][..., 3], cmap="gray")
            c += 1
        axes[r, c].imshow(masks[idx].squeeze(), cmap="gray")
        axes[r, c + 1].imshow((preds[idx] > threshold).astype(np.float32), cmap="gray")
        for a in axes[r]:
            a.axis("off")
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=10, fontweight="bold")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
