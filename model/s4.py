import torch
import torch.nn as nn
from torch.linalg import eigh, inv, matrix_power
import torch.nn.functional as F
import math
torch.set_printoptions(precision=3, sci_mode=False)

def log_step_initializer(dt_min=0.001, dt_max=0.1, shape=(1,)):
    return torch.empty(shape).uniform_(math.log(dt_min), math.log(dt_max))

def make_HiPPO(N):
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    nhippo = make_HiPPO(N)
    P = torch.sqrt(torch.arange(N) + 0.5)
    B = torch.sqrt(2 * torch.arange(N) + 1.0)
    return nhippo, P, B

def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, None] * P[None, :]
    S_diag = torch.diagonal(S)
    A_real = torch.mean(S_diag) * torch.ones_like(S_diag)
    A_imag, V = eigh(S * -1j)

    P = V.conj().T @ P.to(V.dtype)
    B = V.conj().T @ B.to(V.dtype)
    return A_real + 1j * A_imag, P, B, V

def hippo_initializer(dim, N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return Lambda.repeat(dim, 1), P.repeat(dim, 1), B.repeat(dim, 1)
    
def discrete_DPLR(Lambda, P, B, C, step, L):
    # Convert parameters to matrices
    B = B.unsqueeze(-1)
    C = C.unsqueeze(-2)
    P = P.unsqueeze(-1) # dim, N, 1
    Q = P.conj().transpose(-2, -1) # dim, 1, N
    N = Lambda.shape[-1]
    A = torch.diag_embed(Lambda) - P @ Q
    I = torch.eye(N).to(Lambda.device)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = torch.diag_embed(1.0 / ((2.0 / step) - Lambda)) # dim, N, N
    A1 = D - (D @ P * (1.0 / (1 + (Q @ D @ P))) * Q @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = C @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()
    
def scan_SSM(Ab, Bb, Cb, u, x0):
    # u.Size(..., seq, dim)
    x_k, y = x0, []
    Bb = Bb.unsqueeze(0)       # 1,            dim, N, 1
    Cb = Cb.unsqueeze(0)       # 1,            dim, 1, N
    u = u.view(*u.shape, 1, 1) # batch, (seq), dim, 1, 1
    for u_k in u.unbind(dim=-4):
        x_k = Ab @ x_k + Bb @ u_k.to(Bb.dtype)
        y_k = Cb @ x_k
        y.append(y_k.real.squeeze((-2, -1)))
    return torch.stack(y).transpose(0,1), x_k

def cauchy(v, omega, lambd):
    return (v / (omega - lambd)).sum(dim=-1)

def kernel_DPLR(Lambda, P, B, C, step, L):
    Omega_L = torch.exp((-2j * math.pi) * (torch.arange(L) / L)).to(Lambda.device)
    
    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L)).view(L, 1, 1)
    c = 2.0 / (1.0 + Omega_L).view(L, 1)

    aterm = (C.conj(), P.conj())
    bterm = (B, P)

    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10) # seq, dim
    out = torch.fft.ifft(atRoots, L, dim=-2)
    return out.real

def conv(u, K):
    ud = torch.fft.rfft(torch.cat([u, torch.zeros_like(u)], dim=-2), dim=-2)
    Kd = torch.fft.rfft(torch.cat([K, torch.zeros_like(K)], dim=-2), dim=-2)
    return torch.fft.irfft(ud*Kd, dim=-2)[...,:u.shape[-2], :]

class SSM(nn.Module):
    def __init__(self, dim, N):
        super().__init__()
        init_Lambda, init_P, init_B = hippo_initializer(dim, N)
        init_C = torch.randn(dim, N, dtype=torch.cfloat)

        self.Lambda_re, self.Lambda_im = nn.Parameter(init_Lambda.real), nn.Parameter(init_Lambda.imag)
        self.P_re, self.P_im = nn.Parameter(init_P.real), nn.Parameter(init_P.imag)
        self.B_re, self.B_im = nn.Parameter(init_B.real), nn.Parameter(init_B.imag)
        self.C_re, self.C_im = nn.Parameter(init_C.real), nn.Parameter(init_C.imag)
        self.step = nn.Parameter(log_step_initializer(shape=(1,)))
        
        self.register_buffer("x", torch.zeros(dim,N,1, dtype=torch.cfloat))
    def forward(self, u, cnn=True, L=0):
        if L<1: L = u.shape[-2]
        Lambda = self.Lambda_re + 1j * self.Lambda_im
        P = self.P_re + 1j * self.P_im
        B = self.B_re + 1j * self.B_im
        C = self.C_re + 1j * self.C_im
        step = torch.exp(self.step)
        if cnn:
            K = kernel_DPLR(Lambda, P, B, C, step, L)
            return conv(u, K)
        else:
            Ab, Bb, Cb = discrete_DPLR(Lambda, P, B, C, step, L)
            y, self.x = scan_SSM(Ab, Bb, Cb, u, self.x)
            return y.view(u.shape)
            
    def reset_x(self): self.x = torch.zeros_like(self.x)