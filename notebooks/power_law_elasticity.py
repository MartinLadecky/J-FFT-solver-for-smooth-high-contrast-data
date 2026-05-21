# %% [markdown]
# The MIT License (MIT)
# 
# Copyright © 2026 Martin Ladecký, Ivana Pultarová, François Bignonnet, Indre Jödicke, Jan Zeman, Lars Pastewka
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# %% [markdown]
# 
# %%
import numpy as np
import scipy as sc
import scipy.sparse.linalg as sp
import matplotlib  as mpl

import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Utility Functions
# %%
def get_shape_function_gradients(discretization_type, pixel_size):
    """
    Returns the shape function gradients and quadrature weights for a single pixel.
    Currently supports 'linear_triangles'.
    """
    if discretization_type == 'linear_triangles':
        ndim = 2
        nb_quad = 2
        dx, dy = pixel_size
        # B(dim, quad_point, node_idx)
        B = np.zeros([ndim, nb_quad, 4])
        # first quad point
        B[:, 0, :] = [[-1 / dx, 0, 1 / dx, 0],
                      [-1 / dy, 1 / dy, 0, 0]]
        # second quad point
        B[:, 1, :] = [[0, -1 / dx, 0, 1 / dx],
                      [0, 0, -1 / dy, 1 / dy]]

        weights = np.array([dx * dy / 2, dx * dy / 2])
        return B.reshape(ndim, nb_quad, 2, 2), weights

    if discretization_type == 'trilinear_hexahedron':
        #
        #                    ζ
        #                    ^
        #         (-1,1,1)   |     (1,1,1)
        #                7---|------8
        #               /|   |     /|
        #              / |   |    / |
        #   (-1,-1,1) 5----------6  | (1,-1,1)
        #             |  |   |   |  |
        #             |  |   |   |  |
        #             |  |   +---|-------> ξ
        #             |  |  /    |  |
        #   (-1,1,-1) |  3-/-----|--4 (1,1,-1)
        #             | / /      | /
        #             |/ /       |/
        #             1-/--------2
        #   (-1,-1,-1) /        (1,-1,-1)
        #             /
        #            η
        # N(n,i,j,k)
        # N₁ = (1 - ξ) (1 - η) (1 - ζ) / 8  --   N [0,0,0,0]
        # N₂ = (1 + ξ) (1 - η) (1 - ζ) / 8  --   N [0,1,0,0]
        # N₃ = (1 - ξ) (1 + η) (1 - ζ) / 8  --   N [0,0,1,0]
        # N₄ = (1 + ξ) (1 + η) (1 - ζ) / 8  --   N [0,1,1,0]
        # N₅ = (1 - ξ) (1 - η) (1 + ζ) / 8  --   N [0,0,0,1]
        # N₆ = (1 + ξ) (1 - η) (1 + ζ) / 8  --   N [0,1,0,1]
        # N₇ = (1 - ξ) (1 + η) (1 + ζ) / 8  --   N [0,0,1,1]
        # N₈ = (1 + ξ) (1 + η) (1 + ζ) / 8  --   N [0,1,1,1]

        #
        # ∂N₁/∂ξ = - (1 - η) (1 - ζ) / 8  -- B [0,q,0,0,0,0]
        # ∂N₁/∂η = - (1 - ξ) (1 - ζ) / 8  -- B [1,q,0,0,0,0]
        # ∂N₁/∂ζ = - (1 - ξ) (1 - η) / 8  -- B [2,q,0,0,0,0]
        #
        # ∂N₂/∂ξ = + (1 - η) (1 - ζ) / 8  -- B [0,q,0,1,0,0]
        # ∂N₂/∂η = - (1 + ξ) (1 - ζ) / 8  -- B [1,q,0,1,0,0]
        # ∂N₂/∂ζ = - (1 + ξ) (1 - η) / 8  -- B [2,q,0,1,0,0]
        #
        # ∂N₃/∂ξ = - (1 + η) (1 - ζ) / 8  -- B [0,q,0,0,1,0]
        # ∂N₃/∂η = + (1 - ξ) (1 - ζ) / 8  -- B [1,q,0,0,1,0]
        # ∂N₃/∂ζ = - (1 - ξ) (1 + η) / 8  -- B [2,q,0,0,1,0]
        #
        # ∂N₄/∂ξ = + (1 + η) (1 - ζ) / 8  -- B [0,q,0,1,1,0]
        # ∂N₄/∂η = + (1 + ξ) (1 - ζ) / 8  -- B [1,q,0,1,1,0]
        # ∂N₄/∂ζ = - (1 + ξ) (1 + η) / 8  -- B [2,q,0,1,1,0]
        #
        # ∂N₅/∂ξ = - (1 - η) (1 + ζ) / 8  -- B [0,q,0,0,0,1]
        # ∂N₅/∂η = - (1 - ξ) (1 + ζ) / 8  -- B [1,q,0,0,0,1]
        # ∂N₅/∂ζ = + (1 - ξ) (1 - η) / 8  -- B [2,q,0,0,0,1]
        #
        # ∂N₆/∂ξ = + (1 - η) (1 + ζ) / 8  -- B [0,q,0,1,0,1]
        # ∂N₆/∂η = - (1 + ξ) (1 + ζ) / 8  -- B [1,q,0,1,0,1]
        # ∂N₆/∂ζ = + (1 + ξ) (1 - η) / 8  -- B [2,q,0,1,0,1]
        #
        # ∂N₇/∂ξ = - (1 + η) (1 + ζ) / 8  -- B [0,q,0,0,1,1]
        # ∂N₇/∂η = + (1 - ξ) (1 + ζ) / 8  -- B [1,q,0,0,1,1]
        # ∂N₇/∂ζ = + (1 - ξ) (1 + η) / 8  -- B [2,q,0,0,1,1]
        #
        # ∂N₈/∂ξ = + (1 + η) (1 + ζ) / 8  -- B [0,q,0,1,1,1]
        # ∂N₈/∂η = + (1 + ξ) (1 + ζ) / 8  -- B [1,q,0,1,1,1]
        # ∂N₈/∂ζ = + (1 + ξ) (1 + η) / 8  -- B [2,q,0,1,1,1]
        #
        # quad points:
        # ξ1  = -1/√3, η0 = -1/√3, ζ0 = -1/√3
        # ξ2  =  1/√3, η1 = -1/√3, ζ1 = -1/√3
        # ξ3  = -1/√3, η2 =  1/√3, ζ2 = -1/√3
        # ξ4, =  1/√3, η3 =  1/√3, ζ3 = -1/√3
        # ξ5  = -1/√3, η4 = -1/√3, ζ4 =  1/√3
        # ξ6  =  1/√3, η5 = -1/√3, ζ5 =  1/√3
        # ξ7  = -1/√3, η6 =  1/√3, ζ6 =  1/√3
        # Voxel discretization setting
        nb_quad_points_per_pixel = 8
        ndim = 3

        #  pixel sizes for better readability
        del_x, del_y, del_z = pixel_size

        # quadrature points : coordinates
        quad_points_coord = np.zeros([ndim, nb_quad_points_per_pixel])

        coord_helper = np.zeros(2)
        coord_helper[0] = -1. / (np.sqrt(3))
        coord_helper[1] = +1. / (np.sqrt(3))

        # quadrature points    # TODO This hold for prototypical element      !!!
        # TODO MAKE clear how to generate B matrices
        quad_points_coord[:, 0] = [coord_helper[0], coord_helper[0], coord_helper[0]]
        quad_points_coord[:, 1] = [coord_helper[1], coord_helper[0], coord_helper[0]]
        quad_points_coord[:, 2] = [coord_helper[0], coord_helper[1], coord_helper[0]]
        quad_points_coord[:, 3] = [coord_helper[1], coord_helper[1], coord_helper[0]]
        quad_points_coord[:, 4] = [coord_helper[0], coord_helper[0], coord_helper[1]]
        quad_points_coord[:, 5] = [coord_helper[1], coord_helper[0], coord_helper[1]]
        quad_points_coord[:, 6] = [coord_helper[0], coord_helper[1], coord_helper[1]]
        quad_points_coord[:, 7] = [coord_helper[1], coord_helper[1], coord_helper[1]]

        # quadrature points : weights
        weights = np.zeros([nb_quad_points_per_pixel])
        weights[:] = del_x * del_y * del_z / 8

        # Jabobian
        jacoby_matrix = np.array([[(del_x / 2), 0, 0],
                                  [0, (del_y / 2), 0],
                                  [0, 0, (del_z / 2)]])

        inv_jacobian = np.linalg.inv(jacoby_matrix)

        # construction of B matrix
        B_dqnijk = np.zeros([ndim, nb_quad_points_per_pixel, 1, 2, 2, 2])
        for quad_point in range(0, nb_quad_points_per_pixel):
            x_q = quad_points_coord[:, quad_point]
            xi = x_q[0]
            eta = x_q[1]
            zeta = x_q[2]
            # this part have to be hard coded
            # @formatter:off

            B_dqnijk[:, quad_point, 0, 0, 0, 0] = np.array([- (1 - eta) * (1 - zeta) / 8,
                                                            - (1 -  xi) * (1 - zeta) / 8,
                                                            - (1 -  xi) * (1 -  eta) / 8])

            B_dqnijk[:, quad_point, 0, 1, 0, 0] = np.array([+ (1 - eta) * (1 - zeta) / 8,
                                                            - (1 +  xi) * (1 - zeta) / 8,
                                                            - (1 +  xi) * (1 -  eta) / 8])

            B_dqnijk[:, quad_point, 0, 0, 1, 0] = np.array([- (1 + eta) * (1 - zeta) / 8,
                                                            + (1 -  xi) * (1 - zeta) / 8,
                                                            - (1 -  xi) * (1 +  eta) / 8])

            B_dqnijk[:, quad_point, 0, 1, 1, 0] = np.array([+ (1 + eta) * (1 - zeta) / 8,
                                                            + (1 +  xi) * (1 - zeta) / 8,
                                                            - (1 +  xi) * (1 +  eta) / 8])

            B_dqnijk[:, quad_point, 0, 0, 0, 1] = np.array([- (1 - eta) * (1 + zeta) / 8,
                                                            - (1 -  xi) * (1 + zeta) / 8,
                                                            + (1 -  xi) * (1 -  eta) / 8])

            B_dqnijk[:, quad_point, 0, 1, 0, 1] = np.array([+ (1 - eta) * (1 + zeta) / 8,
                                                            - (1 +  xi) * (1 + zeta) / 8,
                                                            + (1 +  xi) * (1 -  eta) / 8])

            B_dqnijk[:, quad_point, 0, 0, 1, 1] = np.array([- (1 + eta) * (1 + zeta) / 8,
                                                            + (1 -  xi) * (1 + zeta) / 8,
                                                            + (1 -  xi) * (1 +  eta) / 8])

            B_dqnijk[:, quad_point, 0, 1, 1, 1] = np.array([+ (1 + eta) * (1 + zeta) / 8,
                                                            + (1 +  xi) * (1 + zeta) / 8,
                                                            + (1 +  xi) * (1 +  eta) / 8])

            # @formatter:on
        # multiplication with inverse of jacobian
        B_dqnijk = np.einsum('dt,tqnijk->dqnijk', inv_jacobian, B_dqnijk)

        return B_dqnijk.reshape(ndim, nb_quad_points_per_pixel, 2, 2, 2), weights
    
    
    
# %%

def get_gradient_operators(pixel_size, N, dofs_per_node):
    """
    Creates and returns the gradient (B) and weighted divergence (Bw_t) operators
    for the given grid and discretization.
    """

    if len(N) == 3:
        ndim = 3
        B_dqijk, weights = get_shape_function_gradients('trilinear_hexahedron', pixel_size)
    else:
        ndim = 2
        B_dqijk, weights = get_shape_function_gradients('linear_triangles', pixel_size)
    nb_quad = weights.size

    def B_op(u_ixyz):
        grad_u_ijqxyz = np.zeros([dofs_per_node, ndim, nb_quad, *N])
        for pixel_node in np.ndindex(*( (2,) * ndim)):
            grad_u_ijqxyz += np.einsum('jq,ixyz...->ijqxyz...',
                                      B_dqijk[(..., *pixel_node)],
                                      np.roll(u_ixyz, -1 * np.array(pixel_node), axis=tuple(range(1, ndim + 1))),
                                      optimize='optimal')
        grad_u_ijqxyz  = (grad_u_ijqxyz + np.swapaxes(grad_u_ijqxyz , 0, 1)) / 2
        return grad_u_ijqxyz

    def Bw_t_op(flux_ijqxyz):
        div_flux_ixyz = np.zeros([dofs_per_node, *N])
        # apply quadrature weights
        flux_weighted = np.einsum('ijq...,q->ijq...', flux_ijqxyz, weights, optimize='optimal')
        for pixel_node in np.ndindex(*( (2,) * ndim)):
            div_fnxyz_pixel_node = np.einsum('jq,ijqxyz...->ixyz...',
                                             B_dqijk[(..., *pixel_node)],
                                             flux_weighted, optimize='optimal')
            div_flux_ixyz += np.roll(div_fnxyz_pixel_node, 1 * np.array(pixel_node), axis=tuple(range(1, ndim + 1)))
        return div_flux_ixyz

    return B_op, Bw_t_op, weights

def solve_sparse(A, b, M=None):
    """
    A simple wrapper around scipy's Conjugate Gradient solver that also counts iterations.
    """
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    x, status = sc.sparse.linalg.cg(A, b, M=M, rtol=1e-5, maxiter=10000, callback=callback)
    return x, status, num_iters

# %% [markdown]
# ## Problem Parameters
# 
# %%
# Problem Parameters
# ------------------
nb_quad_points_per_pixel = 8
ndim = 3  # number of dimensions
N_x = N_y =N_z= 32  # number of voxels
N = (N_x, N_y,N_z)

domain_size = (1, 1, 1)
domain_vol = np.prod(domain_size)
pixel_size = tuple(np.array(domain_size) / np.array(N))

# Degrees of freedom configuration
n_u_dofs = ndim  # 2 for elasticity
ndof = n_u_dofs * np.prod(np.array(N))

displacement_shape = (ndim,) + N
grad_shape = (ndim, ndim, nb_quad_points_per_pixel) + N

# Basic FFT operators and tensor dot products
dot22 = lambda A2, B2: np.einsum('ij...,ij...->...', A2, B2)
dot21 = lambda A, v: np.einsum('ij...,j...->i...', A, v)
ddot42 = lambda A, B: np.einsum('ijkl...,lk...->ij...', A, B)
dyad22 = lambda A2, B2: np.einsum('ij...,kl...->ijkl...', A2, B2)
fft = lambda x: np.fft.fftn(x, [*N])
ifft = lambda x: np.fft.ifftn(x, [*N])

# Initialize gradient operators
B, Bw_t, quadrature_weights = get_gradient_operators(pixel_size, N, n_u_dofs)

# %% [markdown]
# ## Problem Definition
# 
# %%

# identity tensors for elasticity
i = np.eye(ndim)
I4 = np.einsum('il,jk', i, i)
I4rt = np.einsum('ik,jl', i, i)
I4s = (I4 + I4rt) / 2.
II = np.einsum('ij,kl ', i, i)
I = np.einsum('ij,qxyz...->ijqxyz...', i, np.ones((nb_quad_points_per_pixel,) + N))
I4d = (I4s - II / 3.)

def elastic(eps, K, mu):
    # elastic stiffness tensor, and stress response
    C4 = K * II[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis] + 2. * mu * I4d[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    sig = ddot42(C4, eps)
    return sig, C4

def nonlin_elastic(eps, K, sig0, eps0, n):
    epsm = np.einsum('ii...', eps) / 3. # trace / 3
    # epsm * I should broadcast correctly if I is (ndim, ndim, nb_quad, Nx, Ny, Nz)
    epsd = eps - epsm * I
    epseq = np.sqrt(2. / 3. * dot22(epsd, epsd))
    
    # avoid zero division
    mask = epseq > 1e-15
    epseq_safe = np.where(mask, epseq, 1.0)
    
    sig = 3. * K * epsm * I + (2. / 3. * sig0 / (eps0 ** n) * (epseq_safe ** (n - 1.)) * epsd)
    # mask is (nb_quad, Nx, Ny, Nz), sig is (ndim, ndim, nb_quad, Nx, Ny, Nz)
    mask_sig = mask[(np.newaxis,)*2]
    sig = np.where(mask_sig, sig, 3. * K * epsm * I)

    K4_d = 2. / 3. * sig0 / (eps0 ** n) * (
                dyad22(epsd, epsd) * 2. / 3. * (n - 1.) * epseq_safe ** (n - 3.) + epseq_safe ** (n - 1.) * I4d[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis])
    # K4_d is (ndim, ndim, ndim, ndim, nb_quad, Nx, Ny, Nz)
    # mask is (nb_quad, Nx, Ny, Nz)
    # Ensure mask broadcasts to (ndim, ndim, ndim, ndim, nb_quad, Nx, Ny, Nz)
    mask_K4 = mask[(np.newaxis,)*4]
    K4 = K * II[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis] + np.where(mask_K4, K4_d, 0.0)
    #K4 = K * II[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]  + K4_d * (epseq != 0.).astype(float)

    return sig, K4

# Reset phase field
phase = np.zeros(N)
phase[N_x//4:3*N_x//4,N_y//4:3*N_y//4,N_z//4:3*N_z//4] = 1.

# Stiffness parameters
K_inc = 0.2
mu_inc = 0.1
K_matrix = 2.0
sig0_matrix = 0.5
eps0_matrix = 0.01
n_exp = 3.0

def constitutive(eps):
    # phase is 1 for matrix, 0 for inclusion
    sig_P1, K4_P1 = nonlin_elastic(eps, K_matrix, sig0_matrix, eps0_matrix, n_exp)
    sig_P2, K4_P2 = elastic(eps, K_inc, mu_inc)

    # phase is (Nx, Ny, Nz), we need it for (..., Nx, Ny, Nz)
    # Adding np.newaxis at the beginning to help broadcasting
    p = phase[(np.newaxis,)*(len(eps.shape) - len(N))]
    
    sig = (1. - p) * sig_P1 + p * sig_P2
    
    p_K4 = phase[(np.newaxis,)*(len(K4_P1.shape) - len(N))]
    K4 = (1. - p_K4) * K4_P1 + p_K4 * K4_P2

    return sig, K4

# Macroscopic loading
macro_grad_ij = np.zeros([ndim, ndim])
macro_grad_ij[0, 1] = 0.025
macro_grad_ij[1, 0] = 0.025
E_ijqxy = np.zeros(grad_shape)
# Correct way to add macroscopic gradient to all points
for i in range(ndim):
    for j in range(ndim):
        E_ijqxy[i, j, ...] = macro_grad_ij[i, j]

# initialize: strain and stress/tangent
eps = np.copy(E_ijqxy)
sig, K4 = constitutive(eps)

# Define system matrix operator K(u) and RHS vector b
K_fun_I = lambda x: Bw_t(
    ddot42(K4,
           B(x.reshape(displacement_shape)))).reshape(-1)
b_I = -Bw_t(sig).reshape(-1)

# --- Preconditioner (Green's) ---
# Use a reference material for Green's preconditioner

K_0 = 0.2
mu_0 = 0.1
ref_mat_data_ijkl = K_0 * II + 2. * mu_0 * (I4s - 1. / 3. * II)
ref_mat_data_ijkl_field = ref_mat_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]

def get_green_preconditioner(ref_mat):
    K_ref_fun = lambda x: Bw_t(ddot42(ref_mat, B(x.reshape(displacement_shape))))
    G_diag_ijxy = np.zeros([n_u_dofs, n_u_dofs, *N])
    for d in range(n_u_dofs):
        unit_impuls_ixy = np.zeros(displacement_shape)
        unit_impuls_ixy[(d,) + ndim * (0,)] = 1
        G_diag_ijxy[:, d, ...] = K_ref_fun(x=unit_impuls_ixy)
    G_diag_ijxy = np.real(fft(x=G_diag_ijxy))
    reshaped_matrices = G_diag_ijxy.reshape(n_u_dofs, n_u_dofs, -1)
    G_batch = reshaped_matrices.transpose(2, 0, 1)
    G_batch[1:, ...] = np.linalg.inv(G_batch[1:, ...]) 
    M_diag_ijxy = G_batch.transpose(1, 2, 0).reshape(n_u_dofs, n_u_dofs, *N)
    return lambda x: np.real(ifft(dot21(M_diag_ijxy, fft(x=x.reshape(displacement_shape))))).reshape(-1)

Green_fun_I = get_green_preconditioner(ref_mat_data_ijkl_field)

# --- Newton-Raphson Loop ---
En = np.linalg.norm(eps)
iiter = 0
norm_fluctuation = []
nb_iter_CG_per_Newton = []

print(f"Starting Newton-Raphson iterations...")
while True:
    # plot data
    # normalization
    norm = mpl.colors.Normalize(vmin=1, vmax=2500)

    fig, ax = plt.subplots()
    ax.set_title('Newton iteration = {}'.format(iiter))
    im = ax.imshow(K4[0, 1, 0, 1, ..., N_z // 2].mean(axis=0) / 0.1, cmap='viridis', norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\mathrm{C}_{66}/\mu^0$')
    plt.show()

    nb_iter_CG = [0]

    def callback(x):
        #nonlocal nb_iter_CG
        nb_iter_CG[0] += 1

    # Update tangent operator based on current K4
    K_op = sp.LinearOperator((ndof, ndof), 
                              lambda x: Bw_t(ddot42(K4, B(x.reshape(displacement_shape)))).reshape(-1))
    
    # Solve linear system: K_tangent * deps = -RHS
    dx_m, status = sp.cg(K_op, b_I, M=sp.LinearOperator((ndof, ndof), Green_fun_I),
                         rtol=1e-5, maxiter=10000, callback=callback
                         )

    nb_iter_CG_per_Newton.append(nb_iter_CG)
    depsm=B(dx_m.reshape(displacement_shape))
    eps += depsm.reshape(grad_shape)
    
    # Update stress and tangent
    sig, K4 = constitutive(eps)
    b_I = -Bw_t(sig).reshape(-1)
    
    res_norm = np.linalg.norm(depsm) / En
    norm_fluctuation.append(np.linalg.norm(depsm))
    print(f"Newton iter {iiter}: CG iters = {nb_iter_CG}, Residual = {res_norm:.2e}")
    
    if res_norm < 1e-4 and iiter > 0:
        break
    if iiter > 50:
        print("Max Newton iterations reached")
        break
    iiter += 1

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(norm_fluctuation/ En, 'o-')
plt.xlabel('Newton iteration')
plt.ylabel('Norm of update / En')
plt.title('Newton-Raphson Convergence')

plt.subplot(1, 2, 2)
plt.plot(nb_iter_CG_per_Newton, 'o-')
plt.xlabel('Newton iteration')
plt.ylabel('CG iterations')
plt.title('Linear Solver Performance')
plt.show()

# %%
