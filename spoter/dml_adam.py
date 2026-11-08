"""
Adam ที่คำนวณเหมือน torch.optim.Adam (ค่า default: amsgrad=False, maximize=False)
แต่ไม่ใช้ lerp ซึ่ง DirectML ยังไม่รองรับ -> ไม่ต้อง fallback ไป CPU ทุก step

ใช้ใน train.py:
    from dml_adam import DmlAdam
    sgd_optimizer = DmlAdam(slrt_model.parameters(), lr=args.lr)

ทดสอบว่าได้ผลเท่ากับ torch.optim.Adam (รันบน CPU):
    python dml_adam.py
"""
import math

import torch


class DmlAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # m = b1*m + (1-b1)*g   (เทียบเท่า exp_avg.lerp_(grad, 1-b1))
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = b2*v + (1-b2)*g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_c1)
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.nn.Sequential(torch.nn.Linear(20, 30), torch.nn.ReLU(), torch.nn.Linear(30, 5))
    b = torch.nn.Sequential(torch.nn.Linear(20, 30), torch.nn.ReLU(), torch.nn.Linear(30, 5))
    b.load_state_dict(a.state_dict())
    opt_a = torch.optim.Adam(a.parameters(), lr=1e-3)
    opt_b = DmlAdam(b.parameters(), lr=1e-3)
    for _ in range(200):
        x, y = torch.randn(8, 20), torch.randint(0, 5, (8,))
        for m, o in ((a, opt_a), (b, opt_b)):
            o.zero_grad()
            torch.nn.functional.cross_entropy(m(x), y).backward()
            o.step()
    diff = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    print(f"ผลต่างสูงสุดหลัง 200 step: {diff:.2e} ->", "ผ่าน" if diff < 1e-5 else "ไม่ผ่าน")
