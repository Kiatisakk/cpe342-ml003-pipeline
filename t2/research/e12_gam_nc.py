"""E12: spline-GAM with FORWARD LABEL-NOISE CORRECTION (Patrini et al. 2017), trained from scratch in torch.
P(z|x) = softmax(sum_j f_j(x_j))  (additive = correctly specified under class-conditional independence)
P(~y|x) = P(z|x) @ M_eps,  M_eps = symmetric noise matrix, eps learned jointly.
Without the correction the observed posterior 0.01+0.96p is not additive, so a plain GAM is misspecified.
usage: python t2/research/e12_gam_nc.py [n_knots] [l2] [nc 0/1]"""
import sys; sys.path.insert(0, 't2/research')
NK = int(sys.argv[1]) if len(sys.argv) > 1 else 8
L2 = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 1
sys.argv = sys.argv[:1] + [str(NK), '1.0']
from runner import *
import e02_gam as G
import torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))


def fp(a, b):
    Xt, Xe = G.design(a)
    mu, sd = Xt[a].mean(0), Xt[a].std(0) + 1e-6
    X = torch.tensor((Xt[a] - mu) / sd, dtype=torch.float64); yy = torch.tensor(y[a])
    d = X.shape[1]
    W = torch.zeros(d, K, dtype=torch.float64, requires_grad=True); c = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    e = torch.tensor(np.log(0.03 / 0.97), dtype=torch.float64, requires_grad=bool(NC))
    params = [W, c] + ([e] if NC else [])
    opt = torch.optim.LBFGS(params, lr=1, max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-10, history_size=20, line_search_fn='strong_wolfe')
    def Mmat():
        eps = torch.sigmoid(e); return (1 - eps) * torch.eye(K, dtype=torch.float64) + eps / 3 * (1 - torch.eye(K, dtype=torch.float64))
    def closure():
        opt.zero_grad()
        P = torch.softmax(X @ W + c, 1)
        Q = P @ Mmat() if NC else P
        loss = -torch.log(Q[torch.arange(len(yy)), yy] + 1e-12).mean() + L2 * (W ** 2).sum()
        loss.backward(); return loss
    for _ in range(3): opt.step(closure)
    fp.eps.append(float(torch.sigmoid(e)) if NC else 0.0)
    with torch.no_grad():
        M = Mmat().numpy() if NC else np.eye(K)
        pr = lambda Z: softmax(((Z - mu) / sd) @ W.numpy() + c.numpy()) @ M
        return pr(Xt[b]), pr(Xe)


fp.eps = []
if __name__ == '__main__':
    run(f'gamnc_k{NK}_l{L2:g}_nc{NC}', fp)
    print('eps', np.round(fp.eps, 4))
