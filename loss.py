import torch
import torch.nn.functional as F
import utils

class LossFunction(torch.nn.Module):
    def __init__(self, classifier, alpha, lamda, **kwargs):
        super(LossFunction, self).__init__(**kwargs)
        
        self.classifier = classifier
        self.c = alpha
        self.lamda = lamda

    def gradient(self, x):
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top 
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def gradient_loss(self, gt, gen, alpha=1):
        # dx = I(x+1) - I(x), dy = I(y+1) - I(y),
        gen_dx, gen_dy = self.gradient(gen)
        gt_dx, gt_dy = self.gradient(gt)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x ** alpha), torch.mean(grad_diff_y ** alpha)

    def l1_loss(self, a, b):
        return torch.nn.L1Loss()(a, b)

    def l2_loss(self, a, b):
        return torch.nn.MSELoss()(a, b)

    def recosntruction_loss(self, x, z):
        return self.l1_loss(z, x)

    def perceptual_loss(self, real, target, output):
        _, z = self.classifier(output)
        _, t = self.classifier(target)
        _, r = self.classifier(real)

        total_loss = 0

        total_loss += self.l1_loss(z[0], r[0])/(112*112*64)
        total_loss += self.l1_loss(z[1], r[1])/(56*56*256)
        total_loss += self.l1_loss(z[2], r[2])/(28*28*512)
        total_loss += self.l1_loss(z[3], t[3])/(7*7*512)

        total_loss -= (self.lamda*self.l1_loss(z[4], t[4]))

        return total_loss
    
    def m_loss(self, m):
        m_dx, m_dy = self.gradient(m)
        m1 = torch.mean(torch.abs(m))
        m2 = torch.mean(torch.abs(m_dx))
        m3 = torch.mean(torch.abs(m_dy))

        return m1, m2, m3
    
    def forward(self, outputs, targets):
        real, target, y = targets
        raw, mask, masked, yhat = outputs
        l2_loss = self.l2_loss(y, yhat)

        Lr_raw    = self.recosntruction_loss(real, raw)
        Lr_masked = self.recosntruction_loss(real, masked)

        Lp_raw    = self.perceptual_loss(real, target, raw)
        Lp_masked = self.perceptual_loss(real, target, masked)

        Lgx_raw, Lgy_raw       = self.gradient_loss(real, raw)
        Lgx_masked, Lgy_masked = self.gradient_loss(real, masked)

        Lm, Lm_x, Lm_y = self.m_loss(mask)

        total_loss = (self.c[0]*l2_loss +
                      self.c[1]*Lr_raw  + self.c[1]*Lr_masked + 
                      self.c[2]*Lgx_raw + self.c[2]*Lgy_raw   + self.c[2]*Lgx_masked+self.c[2]*Lgy_masked + 
                      self.c[3]*Lp_raw  + self.c[3]*Lp_masked + 
                      self.c[4]*Lm + 
                      self.c[5]*Lm_x + self.c[5]*Lm_y)
        
        return total_loss.view(-1, 1)
#         return total_loss, [l2_loss.data.item(),
#                             Lr_raw.data.item(), Lr_masked.data.item(), 
#                             Lp_raw.data.item(), Lp_masked.data.item(), 
#                             (Lgx_raw+Lgy_raw).data.item(), (Lgy_masked+Lgx_masked).data.item(), 
#                             (Lm+Lm_x+Lm_y).data.item()]

class MSEloss(torch.nn.Module):
    
    def forward(self, outputs, targets):
        raw, mask, masked, yhat = outputs
        real, target, y = targets
        
        return torch.nn.MSELoss()(y, yhat).view(-1, 1)