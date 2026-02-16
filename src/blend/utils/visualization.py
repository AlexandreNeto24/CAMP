import matplotlib.pyplot as plt
import numpy as np

def visualize_results(composite, real, result, mask=None, title="Harmonization Result"):
    """
    Visualiza Composite (Input), Result (Output), Real (Target).
    Args:
        composite (PIL.Image or Tensor): Imagem colada.
        real (PIL.Image or Tensor): Imagem real.
        result (PIL.Image or Tensor): Resultado do modelo.
        mask (PIL.Image or Tensor): Máscara (opcional).
    """
    # Converte Tensores para Numpy se necessário
    def to_np(img):
        if hasattr(img, 'permute'): # Tensor
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
        elif hasattr(img, 'convert'): # PIL
            img = np.array(img)
        return img

    comp_np = to_np(composite)
    real_np = to_np(real)
    res_np = to_np(result)
    
    cols = 4 if mask is not None else 3
    fig, ax = plt.subplots(1, cols, figsize=(5*cols, 5))
    
    ax[0].imshow(comp_np)
    ax[0].set_title("Input (Composite)")
    ax[0].axis('off')
    
    ax[1].imshow(res_np)
    ax[1].set_title("Result (Harmonized)")
    ax[1].axis('off')

    ax[2].imshow(real_np)
    ax[2].set_title("Target (Real)")
    ax[2].axis('off')
    
    if mask is not None:
        mask_np = to_np(mask)
        if mask_np.ndim == 3: mask_np = mask_np.mean(axis=2)
        ax[3].imshow(mask_np, cmap='gray')
        ax[3].set_title("Mask")
        ax[3].axis('off')
        
    plt.suptitle(title)
    plt.tight_layout()
    return fig
