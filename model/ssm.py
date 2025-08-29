import torch
import torch.nn as nn
from torch.linalg import eigh, inv, matrix_power
import torch.nn.functional as F
import math
torch.set_printoptions(precision=3, sci_mode=False)

def discrete_DPLR(Lambda, P, B, C, step, L):
    # Convert parameters to matrices
    #P = P[..., None]
    Q = P.conj().T
    #B = B[..., None]
    #Ct = C[None, :]
    N = Lambda.shape[0]
    #print(f'A = Lambda{torch.diag(Lambda).shape} - P:{P.shape} @ Q:{Q.shape}')
    A = torch.diag(Lambda) - (P @ Q)
    I = torch.eye(N).to(step.device)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = torch.diag(1.0 / ((2.0 / step) - Lambda))
    #Qc = Q.conj().T.reshape(1, -1)
    #P2 = P.reshape(-1, 1)
    #print(f'A1 = D:{(D).shape} @ Q:{(Q).shape} @ P:{(P).shape} @ D:{(D).shape}')
    #A1 = D - (D @ P @ (1.0 / (1 + (Q @ D @ P))) @ Q @ D)
    A1 = D - (D @ P @ inv(torch.eye(Q.shape[0]).to(Q.device) + (Q @ D @ P)) @ Q @ D)

    # A bar and B bar
    #print(f'Ab = A1:{A1.shape} @ A0:{A0.shape}')
    Ab = A1 @ A0
    #print(f'Bb = A1:{A1.shape} @ B:{B.shape}')
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    #print(f'Bb:{Bb.shape}')
    Cb = C @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()

def scan_SSM(Ab, Bb, Cb, u, x):
    assert Ab.shape[0] == Ab.shape[-1]
    x_k, y = x, []
    for idx in range(u.shape[1]):
      #print(f'x_k = Ab:{Ab.shape} @ x:{x[idx].shape} + Bb:{Bb.shape} @ u:{u[:,idx,:].unsqueeze(-1).shape}')
      x_k = Ab @ x_k.to(Ab.device) +  Bb @ u[:, idx, :].unsqueeze(-1).to(Bb.dtype)
      #print(f'y_k = Cb:{Cb.shape} @ x_k:{x_k.shape}')
      y_k = (Cb @ x_k).squeeze(-1)
      y.append(y_k)
      #print(f'y_k{y_k.shape}')
    return torch.stack(y).real.transpose(0, 1), x_k

def make_HiPPO(N):
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = torch.sqrt(torch.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = torch.sqrt(2 * torch.arange(N) + 1.0)
    return nhippo, P, B

def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, None] * P[None, :]

    # Check skew symmetry
    S_diag = torch.diagonal(S)
    A_real = torch.mean(S_diag) * torch.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    A_imag, V = eigh(S * -1j)

    P = V.conj().T @ P.to(V.dtype)
    B = V.conj().T @ B.to(V.dtype)
    return A_real + 1j * A_imag, P, B, V

def hippo_initializer(N):
    return make_DPLR_HiPPO(N)

def log_step_initializer(dt_min=0.001, dt_max=0.1, shape=(1,)):
    return torch.empty(shape).uniform_(math.log(dt_min), math.log(dt_max))

def cauchy(v, omega, lambd):
    #print(f'v:{v.shape}, omega:{omega.shape}, lambd:{lambd.shape}')

    #return (v / (omega.unsqueeze(-1) - lambd.unsqueeze(0))).sum() # 1D
    return torch.stack([(v / (z - lambd)).sum(dim=-1) for z in omega]) # loop
    #return torch.sum(v.unsqueeze(0) / (omega.unsqueeze(-1) - lambd.unsqueeze(0)).unsqueeze(1).unsqueeze(1), dim=-1) # matrix

def kernel_DPLR(Lambda, P, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    #print(f'Lambda:{Lambda.shape}, P:{P.shape}, Q:{Q.shape}, B:{B.shape}, C:{C.shape}, step:{step.shape}, L:{L}')
    Omega_L = torch.exp((-2j * math.pi) * (torch.arange(L) / L)).to(step.device) #Discrete Fourier

    aterm = (C.conj().unsqueeze(-2), P.conj().T.unsqueeze(-2))
    bterm = (B.T.unsqueeze(-3), P.T.unsqueeze(-3))
    #print(aterm[0].shape, aterm[1].shape, bterm[0].shape, bterm[1].shape)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L).view(-1, 1, 1)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    #print(f'k00:{k00.shape},k01:{k01.shape},k10:{k10.shape},k11:{k11.shape},c:{c.shape}')
    #atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    atRoots = c * (k00 - k01 * inv(torch.eye(k11.shape[-1]).to(k11.device) + k11) * k10)
    out = torch.fft.ifft(atRoots, L, dim=0)
    #print(f'out:{out.shape}, atRoots{atRoots.shape}')
    return out.real

def conv(u, K):
    u = u.unsqueeze(-1)
    K = K.unsqueeze(0)
    #print(u.shape, K.shape)
    assert K.shape[1] == u.shape[1]
    ud = torch.fft.rfft(F.pad(u, (*(0, 0), *(0, 0), *(0, K.shape[1]))), dim=1)
    Kd = torch.fft.rfft(F.pad(K, (*(0, 0), *(0, 0), *(0, u.shape[1]))), dim=1)
    #print(ud.shape, Kd.shape)
    #print((Kd @ ud).shape)
    return torch.fft.irfft(Kd @ ud, dim=1)[:, :K.shape[1], ...].squeeze(-1)

class SSM(nn.Module):
  def __init__(self, dim, N, div=1):
    super().__init__()
    assert dim % div == 0
      
    init_Lambda, init_P, init_B, _ = hippo_initializer(N)
    init_P = init_P.unsqueeze(-1).repeat(1,1)
    init_B = init_B.unsqueeze(-1).repeat(1,dim)
    #init_C = torch.randn(2, dim, N) * (0.5**0.5)

    #vmap
    init_Lambda = init_Lambda.unsqueeze(-1).repeat(1, div)
    init_P = init_P.unsqueeze(-1).repeat(1,1, div)
    init_B = init_B.view(init_B.shape[0], init_B.shape[1]//div, div)
    init_C = torch.randn(2, dim//div, N, div) * (0.5**0.5)

    self.Lambda_re, self.Lambda_im = nn.Parameter(init_Lambda.real), nn.Parameter(init_Lambda.imag)
    self.P_re, self.P_im = nn.Parameter(init_P.real), nn.Parameter(init_P.imag)
    self.B_re, self.B_im = nn.Parameter(init_B.real), nn.Parameter(init_B.imag)
    self.C_re, self.C_im = nn.Parameter(init_C[0]), nn.Parameter(init_C[1])
    #self.D = nn.Parameter(torch.ones(dim, dim))
    self.step = nn.Parameter(log_step_initializer(shape=(1,)))
    self.div = div

    self.x = torch.zeros(1, N, 1, div, dtype=torch.cfloat)
  def forward(self, u, cnn=True):
    with torch.no_grad():
      self.Lambda_re.clamp_(min=None, max=-1e-4)
    Lambda = (self.Lambda_re + 1j * self.Lambda_im)
    P = (self.P_re + 1j * self.P_im)
    B = (self.B_re + 1j * self.B_im)
    C = (self.C_re + 1j * self.C_im)
    b, L, d = u.shape
    u = u.view(b, L, d//self.div, self.div)
    step = torch.exp(self.step)

    #print(f'Lambda:{Lambda.shape}, P:{P.shape}, B:{B.shape}, C:{C.shape}, step:{step.shape}')
    if cnn:
      K = torch.vmap(kernel_DPLR, in_dims=(-1, -1, -1, -1, None, None), out_dims=-1)(Lambda, P, B, C, step, L)
      #print(u.shape, K.shape)
      #return conv(u, K)# + x @ self.D
      return torch.vmap(conv, in_dims=(-1, -1), out_dims=-1)(u, K).view(b, L, d)
    else:
      Ab, Bb, Cb = torch.vmap(discrete_DPLR, in_dims=(-1, -1, -1, -1, None, None), out_dims=-1)(Lambda, P, B, C, step, L)
      y, self.x = torch.vmap(scan_SSM, in_dims=(-1, -1, -1, -1, -1), out_dims=-1)(Ab, Bb, Cb, u, self.x)
      #print(self.x[0, :1, 0])
      return y.view(b, L, d)