import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    """Foca-se nas áreas da máscara para harmonizar"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class DoveNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (Downsampling Contextual)
        # Input: RGB(3) + Mask(1) + Boundary(1) = 5 Canais
        self.conv1 = BasicConv(5, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = BasicConv(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck com Dilated Convolutions (Global Domain Context)
        self.bridge = nn.Sequential(
            BasicConv(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=4, dilation=4),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1)
        )
        
        # Attention Gate
        self.sam = SpatialAttention()
        
        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv4 = BasicConv(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv3 = BasicConv(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv2 = BasicConv(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        
    def forward(self, image, mask, boundary_mask):
        """
        image: [B, 3, H, W] - Composite
        mask: [B, 1, H, W] - Object Mask Hard
        boundary_mask: [B, 1, H, W] - Soft Mask (1.0 borda, 0.1 centro)
        """
        # Input: Concatena tudo para o modelo saber ONDE focar e ONDE proteger
        x = torch.cat([image, mask, boundary_mask], dim=1)
        
        # Encoder
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        
        # Bridge
        bridge = self.bridge(c4)
        
        # Decoder
        u4 = self.up4(bridge)
        u4 = torch.cat([u4, c3], dim=1)
        d4 = self.dconv4(u4)
        
        u3 = self.up3(d4)
        u3 = torch.cat([u3, c2], dim=1)
        d3 = self.dconv3(u3)
        
        u2 = self.up2(d3)
        u2 = torch.cat([u2, c1], dim=1)
        d2 = self.dconv2(u2)
        
        # Attention Refinement
        att_map = self.sam(d2)
        d2 = d2 * att_map + d2 
        
        # Prediction (Delta)
        delta = self.final_conv(d2)
        
        # Prediction Bruta (Imagem Totalmente Harmonizada)
        harmonized_full = torch.clamp(image + delta, 0, 1)
        
        # --- MIXING STAGE ---
        # Suavizar a boundary_mask para eliminar degraus na transição.
        # A mask original (0.0/0.1/1.0 discretos) é boa para GUIAR o encoder,
        # mas para MISTURAR o output precisamos de gradiente contínuo para
        # evitar a borda acinzentada no output.
        smooth_boundary = F.avg_pool2d(boundary_mask, kernel_size=15, stride=1, padding=7)
        # Segundo passe para suavização extra
        smooth_boundary = F.avg_pool2d(smooth_boundary, kernel_size=11, stride=1, padding=5)
        
        # Fórmula de mistura: agora com gradiente suave
        # Onde smooth_boundary ≈ 1 (anel central): usa harmonizado
        # Onde smooth_boundary ≈ 0 (fundo): preserva original
        # Transição contínua entre ambos → sem seam visível
        final_output = (harmonized_full * smooth_boundary) + (image * (1 - smooth_boundary))
        
        return final_output
    
    
class GradientLoss(nn.Module):
    """
    Penaliza diferenças nas DERIVADAS da imagem (edge smoothness).
    Simula o efeito do Poisson Blending.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Filtros de Sobel
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Aplica em cada canal RGB separadamente
        b, c, h, w = pred.shape
        preds_x = F.conv2d(pred.view(b*c, 1, h, w), self.kernel_x, padding=1)
        preds_y = F.conv2d(pred.view(b*c, 1, h, w), self.kernel_y, padding=1)
        
        target_x = F.conv2d(target.view(b*c, 1, h, w), self.kernel_x, padding=1)
        target_y = F.conv2d(target.view(b*c, 1, h, w), self.kernel_y, padding=1)
        
        return self.mse(abs(preds_x), abs(target_x)) + self.mse(abs(preds_y), abs(target_y))
