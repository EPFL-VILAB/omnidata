import matplotlib.pyplot as plt

def show_batch_images(batch, batch_idx, view_idxs=None, keys=('rgb', 'depth_euclidean'), figsize=None):
    if 'positive' in batch: batch = batch['positive']
    if view_idxs is None: view_idxs = list(range(batch[keys[0]].shape[1]))
    max_view_idx = max(view_idxs)
    for k in keys:
        n_batch, n_view = batch[k].shape[:2]
        if n_batch <= batch_idx: raise ValueError(f"Trying to show batch_idx {batch_idx} but batch key {k} is of shape {batch[k].shape}")
        if n_view <= max_view_idx: raise ValueError(f"Trying to show view number {max_view_idx} but batch key {k} is of shape {batch[k].shape}")
    n_rows = len(keys)
    n_cols = len(view_idxs)
    cur_idx = 0
    figsize = figsize if figsize is not None else (4*n_cols, 4*n_rows)
    plt.figure(figsize=figsize)
    for row, key in enumerate(keys):
        for col, view_idx in enumerate(view_idxs):
            cur_idx += 1
            plt.subplot(n_rows, n_cols, cur_idx);
            if row == 0: plt.title(f'View {view_idx}')
            plt.imshow(batch[key][batch_idx][view_idx].permute([1,2,0]));
    plt.tight_layout()
    plt.show()