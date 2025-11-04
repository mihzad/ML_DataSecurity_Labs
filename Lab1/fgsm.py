import torch
import torch.nn.functional as F

# ImageNet mean/std (pixel-space)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

def batch_fgsm_attack(
    model,
    imgs,
    labels,
    epsilon=8/255.0,
    device='cuda',
    targeted=False,
    reduction='sum',
):
    """
    Produce a batch FGSM adversarial batch and RETURN the *normalized* adversarial images.
    - model: torch.nn.Module (expecting normalized images)
    - imgs: tensor [B,3,H,W] in [0,1], dtype float32
    - labels: tensor [B] (long) with true labels (untargeted) or target labels (targeted)
    - epsilon: pixel-space epsilon (e.g. 8/255)
    - targeted: if True perform targeted FGSM (move toward labels), else untargeted
    - reduction: 'sum' or 'mean' for loss reduction (sum recommended for stable gradients)
    Returns:
    - x_adv_norm: tensor [B,3,H,W] already normalized (same device as model/device),
                  dtype float32, detached (ready to pass to model)
    Notes:
    - The returned tensor is in normalized space: (x_adv - mean) / std, so feed directly to model.
    """

    model.eval()
    device = torch.device(device)
    imgs = imgs.to(device)
    labels = labels.to(device, dtype=torch.long)

    # prepare mean/std on device and shaped for broadcasting
    mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(device).view(1, 3, 1, 1)

    # normalize input for gradient computation
    x_norm = (imgs - mean) / std
    x_norm = x_norm.clone().detach().requires_grad_(True)

    # forward pass
    logits = model(x_norm)
    loss = F.cross_entropy(logits, labels, reduction=reduction)
    if targeted:
        loss = -loss


    assert x_norm.requires_grad == True
    assert logits.requires_grad == True

    # backward
    model.zero_grad()
    loss.backward()

    # gradient wrt normalized input
    grad = x_norm.grad.detach()  # [B,3,H,W]

    # convert epsilon (pixel-space) -> normalized-space per channel
    eps_norm = (epsilon / IMAGENET_STD).to(device).view(1, 3, 1, 1)

    # FGSM perturbation (L-inf)
    perturb = eps_norm * grad.sign()

    # apply perturbation in normalized domain
    x_adv_norm = x_norm + perturb

    # clamp in normalized space so corresponding pixels are in [0,1]
    x_min = ((0.0 - mean) / std)
    x_max = ((1.0 - mean) / std)
    x_adv_norm = torch.max(torch.min(x_adv_norm, x_max), x_min)

    # return normalized adv batch, detached and ready to pass to model
    return x_adv_norm.detach()

def single_fgsm_attack(
    model,
    img,
    label,
    epsilon=8/255.0,
    device='cuda',
    targeted=False,
    reduction='sum',
):
    """
    Single-image FGSM (returns the NORMALIZED adversarial image).
    - model: torch.nn.Module expecting normalized inputs
    - img: torch.Tensor [3,H,W], values in [0,1], dtype float32
    - label: int or torch.LongTensor scalar (true class for untargeted)
    - epsilon: pixel-space epsilon (e.g. 8/255)
    - device: device string or torch.device
    - targeted: if True performs targeted FGSM (move toward label)
    - reduction: 'sum' or 'mean' for the per-batch loss reduction (single image -> irrelevant but kept)
    Returns:
    - adv_norm_img: torch.Tensor [3,H,W] in normalized space (i.e., (x-mean)/std), detached
      — ready to pass directly to `model(...)`.
    Notes:
    - The function temporarily modifies the model.train/eval mode if needed, but restores original mode.
    - Ensure `img` is in [0,1]. If coming from a DataLoader that already normalized, do NOT use this function.
    """

    device = torch.device(device)
    model = model.to(device)

    # prepare image and label
    if not torch.is_floating_point(img):
        img = img.float()
    img = img.to(device)
    if not torch.is_tensor(label):
        label_t = torch.tensor([int(label)], device=device, dtype=torch.long)
    else:
        # allow label scalar tensor or 0-d/1-d tensor
        label_t = label.to(device, dtype=torch.long).view(1)

    # mean/std on device, broadcastable
    mean = IMAGENET_MEAN.to(device).view(1,3,1,1)
    std  = IMAGENET_STD.to(device).view(1,3,1,1)

    # normalized input (batch dim added)
    x_norm = (img.unsqueeze(0) - mean) / std
    x_norm = x_norm.clone().detach().requires_grad_(True)

    # keep original mode and set eval to be safe (model.eval() doesn't disable autograd)
    orig_mode = model.training
    model.eval()

    # forward - if output detached, try model.train() briefly (some models/hooks may detach in eval)
    logits = model(x_norm)

    assert x_norm.requires_grad == True
    assert logits.requires_grad == True

    #if not logits.requires_grad:
    #    model.train()
    #    logits = model(x_norm)
    #    # if still detached, raise informative error
    #    if not logits.requires_grad:
    #        # restore original mode
    #        model.train(orig_mode)
    #        raise RuntimeError("Model output is detached (logits.requires_grad is False). "
    #                           "Check model implementation or ensure gradients are enabled.")

    # compute loss for this single example
    loss = F.cross_entropy(logits, label_t, reduction=reduction)
    if targeted:
        loss = -loss

    # zero grads, backward
    model.zero_grad(set_to_none=True)
    loss.backward()

    # gradient w.r.t normalized input
    grad = x_norm.grad
    if grad is None:
        model.train(orig_mode)
        raise RuntimeError("x_norm.grad is None. Ensure x_norm.requires_grad_(True) and autograd is enabled.")

    # convert epsilon (pixel-space) -> normalized-space per channel
    eps_norm = (epsilon / IMAGENET_STD).to(device).view(1,3,1,1)

    # FGSM perturbation in normalized space (L-inf)
    perturb = eps_norm * grad.sign()

    x_adv_norm = x_norm + perturb

    # clamp normalized space so corresponding pixel-space values are in [0,1]
    x_min = ((0.0 - mean) / std)
    x_max = ((1.0 - mean) / std)
    x_adv_norm = torch.max(torch.min(x_adv_norm, x_max), x_min)

    # restore model mode
    model.train(orig_mode)

    # return single image normalized, detached, shaped [3,H,W]
    return x_adv_norm.squeeze(0).detach()
