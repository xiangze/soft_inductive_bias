import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from  train_various_data_net import simple_train

# ===== 2) モデル（ResNet-18 を 10クラスに調整）=====
def make_model():
    model = models.resnet18(weights=None)   # 事前学習なし。ImageNet重みを使うなら weights='IMAGENET1K_V1' など
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def hvp(model, loss_fn, data_loader, v, device='cuda', num_batches=1):
    """
    Hv を返す。データは num_batches バッチで近似平均。
    v は「フラット化したパラメタ」と同じ形の 1D テンソル。
    """
    model.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    # v をパラメタ形状に分割
    shapes = [p.shape for p in params]
    sizes  = [p.numel() for p in params]
    splits = list(v.split(sizes))
    v_params = [s.view(sh) for s, sh in zip(splits, shapes)]

    Hv = [torch.zeros_like(p, device=device) for p in params]
    count = 0

    for (xb, yb) in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        # 一階勾配
        grads = torch.autograd.grad(loss, params, create_graph=True)
        # v に沿った方向微分（ベクトル・ヤコビアン積）
        grad_v = torch.autograd.grad(
            grads, params, grad_outputs=v_params, retain_graph=False
        )
        # 逐次平均
        for i, g in enumerate(grad_v):
            Hv[i] += g.detach()
        count += 1
        if count >= num_batches:
            break

    Hv = [h / count for h in Hv]
    return torch.cat([h.reshape(-1) for h in Hv])

def getvec(device,model,loss_fn,train_loader):
    # パラメタ次元
    dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 上位固有対
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)   # 総パラメタ数
    print("Total dim:", dim)
    v = torch.randn(dim, device=device)  
    v = v / v.norm()                     
    return hvp(model, loss_fn, train_loader, v, device='cuda', num_batches=1)

def getHessianrank_hpv(device,outfilename):
    model,loss_fn,[train_loader,test_loader]=simple_train(device)
    v=getvec(device,model,loss_fn,train_loader)
    with open(outfilename,"w") as fpw:
        print("Hv",v,file=fpw)

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.manual_seed(0)
    getHessianrank_hpv(device,"hpv_result.log")

