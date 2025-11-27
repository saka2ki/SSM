import torch
import torch.nn as nn
from torch.linalg import eigh, inv, matrix_power
import torch.nn.functional as F
import math
torch.set_printoptions(precision=3, sci_mode=False)

def log_step_initializer(dt_min=0.001, dt_max=0.1, shape=(1,)):
    return torch.empty(shape).uniform_(math.log(dt_min), math.log(dt_max))
    
def discrete_DPLR(Lambda, B, C, step, L):
    B = B.unsqueeze(-1)
    C = C.unsqueeze(-2)
    N = Lambda.shape[-1]
    A = torch.diag_embed(Lambda)
    I = torch.eye(N).to(Lambda.device)

    A0 = (2.0 / step) * I + A

    A1 = torch.diag_embed(1.0 / ((2.0 / step) - Lambda)) # dim, N, N

    Ab = A1 @ A0
    Bb = 2 * A1 @ B
    
    Cb = C @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()
    
def scan_SSM(Ab, Bb, Cb, u, x0):
    # u.Size(..., seq, dim)
    _, L, _ = u.shape
    x_k, y = x0, []
    u = u.view(*u.shape, 1, 1) # batch, (seq), dim, 1, 1
    for k in range(L):
        x_k = Ab[:, k, :, :] @ x_k.to(Ab.device) + Bb[:, k, :, :] @ u[:, k, :, :].to(Bb.dtype)
        y_k = Cb[:, k, :, :] @ x_k
        y.append(y_k.real.squeeze((-2, -1)))
    return torch.stack(y).transpose(0,1), x_k

def cauchy(v, omega, lambd):
    return (v / (omega - lambd)).sum(dim=-1)

def kernel_DPLR(Lambda, B, C, step, L):
    Omega_L = torch.exp((-2j * math.pi) * (torch.arange(L) / L)).to(Lambda.device)
    
    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L)).view(L, 1, 1)
    c = 2.0 / (1.0 + Omega_L).view(L, 1)

    atRoots = c * cauchy(C.conj() * B, g, Lambda) # seq, dim
    out = torch.fft.ifft(atRoots, L, dim=-2)
    return out.real

def conv(u, K):
    ud = torch.fft.rfft(torch.cat([u, torch.zeros_like(u)], dim=-2), dim=-2)
    Kd = torch.fft.rfft(torch.cat([K, torch.zeros_like(K)], dim=-2), dim=-2)
    return torch.fft.irfft(ud*Kd, dim=-2)[...,:u.shape[-2], :]

class SSM(nn.Module):
    def __init__(self, dim, N):
        super().__init__()
        init_B = torch.randn(dim, N, dtype=torch.cfloat)
        init_C = torch.randn(dim, N, dtype=torch.cfloat)

        self.B_re, self.B_im = nn.Parameter(init_B.real), nn.Parameter(init_B.imag)
        self.C_re, self.C_im = nn.Parameter(init_C.real), nn.Parameter(init_C.imag)
        self.step = nn.Parameter(log_step_initializer(shape=(1,)))
        
        self.register_buffer("x", torch.zeros(dim,N,1, dtype=torch.cfloat))
    def forward(self, u, cnn=True, L=0):
        if L<1: L = u.shape[-2]
        B_re = torch.sigmoid(torch.einsum('bld,dn->bldn', u, self.B_re))
        B_im = torch.sigmoid(torch.einsum('bld,dn->bldn', u, self.B_im))
        C_re = torch.einsum('bld,dn->bldn', u, self.C_re)
        C_im = torch.einsum('bld,dn->bldn', u, self.C_im)
        Lambda = (1 - B_re) + 1j * (1 - B_im)
        B = B_re + 1j * B_im
        C = C_re + 1j * C_im
        step = torch.exp(self.step)
        if cnn:
            K = kernel_DPLR(Lambda, B, C, step, L)
            return conv(u, K)
        else:
            Ab, Bb, Cb = discrete_DPLR(Lambda, B, C, step, L)
            y, self.x = scan_SSM(Ab, Bb, Cb, u, self.x)
            return y.view(u.shape)
            
    def reset_x(self): self.x = torch.zeros_like(self.x)