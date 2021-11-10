import torch
from torch import nn
import torch.nn.functional as F


class DFIM(nn.Module): 
    def __init__(self, list_k, k, size_id, modes=3):
        super(DFIM, self).__init__()
        self.len = len(list_k)
        self.size_id = size_id
        up = []
        for i in range(len(list_k)):
            up.append(nn.Sequential(nn.Conv2d(list_k[i], k, 1, 1, bias=False), gn(k)))
        self.merge = nn.ModuleList(up)
        merge_convs, fcs, convs = [], [], []
        for m in range(modes):
            merge_convs.append(nn.Sequential(
                        nn.Conv2d(k, k//4, 1, 1, bias=False), 
                        gn(k//4), 
                        nn.ReLU(inplace=True),
                        nn.Conv2d(k//4, k, 1, 1, bias=False),
                        gn(k),
                    ))
            fcs.append(nn.Sequential(
                    nn.Linear(k, k // 4, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(k // 4, self.len, bias=False),
                ))
            convs.append(nn.Sequential(nn.Conv2d(k, k, 3, 1, 1, bias=False), gn(k), nn.ReLU(inplace=True)))
        self.merge_convs = nn.ModuleList(merge_convs)
        self.fcs = nn.ModuleList(fcs)
        self.convs = nn.ModuleList(convs)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.relu =nn.ReLU(inplace=True)

    def forward(self, list_x, mode=3):
        x_size = list_x[self.size_id].size()
        feas = []
        for i in range(len(list_x)):
            feas.append(self.merge[i](F.interpolate(list_x[i], x_size[2:], mode='bilinear', align_corners=True)).unsqueeze(dim=1))
        feas = torch.cat(feas, dim=1) # Nx6xCxHxW
        fea_sum = torch.sum(feas, dim=1) # NxCxHxW

        if mode == 3:
            outs = []
            for mode_ in range(3):
                fea_u = self.merge_convs[mode_](fea_sum)
                fea_s = self.gap(fea_u).squeeze(-1).squeeze(-1) # NxC
                fea_z = self.fcs[mode_](fea_s) # Nx6
                selects = self.softmax(fea_z) # Nx6
                feas_f = selects.reshape(x_size[0], self.len, 1, 1, 1).expand_as(feas) * feas # Nx6xCxHxW
                _, index = torch.topk(selects, 3, dim=1) # Nx3
                selected = []
                for i in range(x_size[0]):
                    selected.append(torch.index_select(feas_f, dim=1, index=index[i]))
                selected = torch.cat(selected, dim=0)
                fea_v = selected.sum(dim=1)
                outs.append(self.convs[mode_](self.relu(fea_v)))
            return torch.cat(outs, dim=0)
        else:
            fea_u = self.merge_convs[mode](fea_sum)
            fea_s = self.gap(fea_u).squeeze(-1).squeeze(-1) # NxC
            fea_z = self.fcs[mode](fea_s) # Nx6
            selects = self.softmax(fea_z) # Nx6
            feas_f = selects.reshape(x_size[0], self.len, 1, 1, 1).expand_as(feas) * feas # Nx6xCxHxW
            _, index = torch.topk(selects, 3, dim=1) # Nx3
            selected = []
            for i in range(x_size[0]):
                selected.append(torch.index_select(feas_f, dim=1, index=index[i]))
            selected = torch.cat(selected, dim=0)
            fea_v = selected.sum(dim=1)
            return self.convs[mode](self.relu(fea_v))


