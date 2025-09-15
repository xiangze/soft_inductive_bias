import torch
from train_various_data_net import simple_train
import hpv

def lanczos_topk(model, loss_fn, data_loader, dim, topk=10, m=80, device='cuda'):
    """
    k 本の上位固有値を Lanczos法 で近似（m は反復長）。
    返り値: evals[k], evecs[dim, k]
    """
    Q = []
    alpha = []
    beta = []

    q = torch.randn(dim, device=device); q /= q.norm()
    Q.append(q)
    b_prev = 0.0

    for j in range(m):
        z = hpv.hvp(model, loss_fn, data_loader, Q[-1], device=device)
        a = torch.dot(Q[-1], z).item()
        alpha.append(a)
        if j == 0:
            r = z - a * Q[-1]
        else:
            r = z - a * Q[-1] - b_prev * Q[-2]

        # 再直交化（数値安定用に1回だけ実施）
        for q_old in Q:
            r -= torch.dot(r, q_old) * q_old

        b = r.norm().item()
        beta.append(b)
        if b < 1e-10:  # 収束
            break
        q = r / b
        Q.append(q)
        b_prev = b

    T = torch.zeros(len(Q), len(Q), device=device)
    for i in range(len(alpha)):
        T[i, i] = alpha[i]
    for i in range(len(beta)-1):
        T[i, i+1] = beta[i+1]
        T[i+1, i] = beta[i+1]

    evals, U = torch.linalg.eigh(T)  # tri-diagonal の固有分解
    idx = torch.argsort(evals, descending=True)[:topk]
    evals = evals[idx]
    U = U[:, idx]

    Qmat = torch.stack(Q, dim=1)        # [dim, m_eff]
    evecs = Qmat @ U                    # Ritz ベクトル [dim, k]
    return evals, evecs

def _get_eigenvecs(device,model,loss_fn,data_loader,topk=5,m=10,tol = 1e-6):
    param_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # top-k eigen values/vectors
    evals, evecs = lanczos_topk(model, loss_fn, data_loader, param_dim, topk=topk, m=m, device=device)
    # ランク推定（しきい値 tol でカウント）,スケールに応じて調整（例: loss/バッチ平均の規模）
    rank_est = int((evals.abs() > tol).sum().item())
    evecs = evecs.to('cpu').detach().numpy().copy()
    evals = evals.to('cpu').detach().numpy().copy()
    return rank_est,evals,evecs

def get_eigenvecs(outfilename,device,topk=5,m=10):
    model,loss_fn,[data_loader,test_loader] =simple_train(device,epochs=100)
    rank_est,evals,evecs= _get_eigenvecs(device,model,loss_fn,data_loader,topk,m)
    
    with open(f"{outfilename}_{m}_{topk}","w") as fpw:
        print(f"反復{m}回",file=fpw)
        print(f"上位{topk}個",file=fpw)
        print(f"rank {rank_est}",file=fpw)
        print(f"固有値{evals}",file=fpw)
        print(f"固有ベクトル{evecs}",file=fpw)


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    get_eigenvecs("lanclog.txt",device,m=5)
