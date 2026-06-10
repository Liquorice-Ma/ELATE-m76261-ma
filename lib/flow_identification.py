import torch


def identify_elephant_mice(F, lam=5.0):
    """Separate traffic matrix into elephant and mice flows (Algorithm 1).

    Args:
        F: traffic matrix tensor of shape (|V|, |V|)
        lam: λ hyperparameter controlling threshold (default=5, per paper Section IV.B)

    Returns:
        F_e: elephant flow matrix (same shape, zeros for mice)
        F_m: mice flow matrix (same shape, zeros for elephants)
        elephant_mask: boolean mask of elephant flows (|V|, |V|)
    """
    mask = ~torch.eye(F.shape[0], dtype=torch.bool, device=F.device)
    values = F[mask]

    mu = values.mean()
    sigma = values.std()
    tau = mu + lam * sigma

    elephant_mask = (F >= tau) & mask
    mice_mask = (F < tau) & mask

    F_e = torch.zeros_like(F)
    F_m = torch.zeros_like(F)
    F_e[elephant_mask] = F[elephant_mask]
    F_m[mice_mask] = F[mice_mask]

    return F_e, F_m, elephant_mask
