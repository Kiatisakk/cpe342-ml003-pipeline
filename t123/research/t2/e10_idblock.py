"""E10 (ID LEAK - see REPORT.md): player_id numbers were assigned in segment blocks
(0..51999 -> 0, 52000..84499 -> 1, 84500..110499 -> 2, 110500..129999 -> 3, i.e. 40/25/20/15% of 130k),
and ~3% of train labels are uniform random flips. Test ids interleave with train ids in the same range.

Model: learn the 3 cut points on the outer-train rows only (max accuracy step function), predict the block class.
Inside the uncertainty gap around a cut (between the last train id of block k and the first id of block k+1 seen
in the outer-train rows) the two adjacent classes are decided by the baseline NB restricted to those 2 classes.
Same 5-fold x 3 seeds scheme. Writes P_idblock_s{seed}.npz like the other runners."""
import sys; sys.path.insert(0, 't123/research/t2')
from runner import *
ID = tr.player_id.str[1:].astype(int).values
IDT = te.player_id.str[1:].astype(int).values


def cuts(a):
    o = a[np.argsort(ID[a])]; ids, yy = ID[o], y[o]; cs, gaps = [], []
    for k in range(K - 1):
        # objective(c) = #(y==k, id<c) + #(y==k+1, id>=c); evaluate at every candidate position
        gain = np.cumsum((yy == k).astype(int) - (yy == k + 1).astype(int))
        i = int(np.argmax(gain))            # cut after position i
        lo, hi = ids[i], ids[i + 1]
        cs.append((lo + hi) / 2); gaps.append((lo, hi))
    return np.array(cs), gaps


NBS = {}


def nb_scores_for(seed):
    if seed not in NBS:
        oof, te_ = nb_blocks(CFG_OPT, 1.0, seed, 'opt'); NBS[seed] = (oof.sum(1), te_.sum(1))
    return NBS[seed]


def make_fp(seed):
    S_oof, S_te = nb_scores_for(seed)
    def fp(a, b):
        cs, gaps = cuts(a)
        def pred(ids, S):
            blk = np.digitize(ids, cs); P = np.full((len(ids), K), 0.01); P[np.arange(len(ids)), blk] = 0.97
            for k, (lo, hi) in enumerate(gaps):      # ambiguous ids strictly inside the gap: use NB on {k,k+1}
                g = (ids > lo) & (ids < hi)
                if g.any():
                    s2 = softmax(S[g][:, [k, k + 1]]); P[g] = 0.01; P[g, k] = 0.01 + 0.96 * s2[:, 0]; P[g, k + 1] = 0.01 + 0.96 * s2[:, 1]
            return P
        return pred(ID[b], S_oof[b]), pred(IDT, S_te)
    return fp


if __name__ == '__main__':
    for s in SEEDS:
        run('idblock', make_fp(s), seeds=(s,))
    summary('idblock')
    print('cuts on full train:', cuts(np.arange(N)))
