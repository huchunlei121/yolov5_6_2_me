# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    """用在ComputeLoss类中
    标签平滑操作  [1, 0]  =>  [0.95, 0.05]
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    :params eps: 平滑参数
    :return positive, negative label smoothing BCE targets  两个值分别代表正样本和负样本的标签取值
            原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        '''
        定义分类损失和置信度损失为带sigmoid的二值交叉熵损失，
        即会先将输入进行sigmoid再计算BinaryCrossEntropyLoss(BCELoss)。
        pos_weight参数是正样本损失的权重参数。
        '''
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # 标签平滑的策略(trick)，是一种在 分类/检测 问题中，防止过拟合的方法 https://blog.csdn.net/qq_38253797/article/details/116228065
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets cp是positive标签 cn是negative标签

        # Focal loss 超参数文件中是为0的
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        '''
        每一层预测值所占的权重比，分别代表浅层到深层，小特征到大特征，4.0对应着P3，1.0对应P4,0.4对应P5。
        如果是自己设置的输出不是3层，则返回[4.0, 1.0, 0.25, 0.06, .02]，可对应1-5个输出层P3-P7的情况。
        '''
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        '''
        autobalance 默认为 False，yolov5中目前也没有使用 ssi = 0即可
        '''
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        '''
        赋值各种参数,gr是用来设置IoU的值在objectness loss中做标签的系数, 
        使用代码如下：
        tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
        train.py源码中model.gr=1，也就是说完全使用标签框与预测框的CIoU值来作为该预测框的objectness标签。
        '''
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of out layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        '''
        na = 3,表示每个预测层anchors的个数
        targets 为一个batch中所有的标签，包括标签所属的image，以及class,x,y,w,h
        targets = [[image1,class1,x1,y1,w1,h1],
                   [image2,class2,x2,y2,w2,h2],
                   ...
                   [imageN,classN,xN,yN,wN,hN]]
        nt为一个batch中所有标签的数量
        '''
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        '''
        gain是为了最终将坐标所属grid坐标限制在坐标系内，不要超出范围,
        其中7是为了对应: image class x y w h ai,
        但后续代码只对x y w h赋值，x,y,w,h = nx,ny,nx,ny,
        nx和ny为当前输出层的grid大小。
        '''
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        '''
        ai.shape = [na,nt]
        ai = [[0,0,0,.....],
              [1,1,1,...],
              [2,2,2,...]]
        这么做的目的是为了给targets增加一个属性，即当前标签所属的anchor索引
        '''
        # ai = torch.zeros(na, nt)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        '''
        tragets.shape = [nt, 6]
        targets.repeat(na, 1, 1).shape = [na,nt,6]
        ai[:, :, None].shape = [na,nt,1](None在list中的作用就是在插入维度1)
        ai[:, :, None] = [[[0],[0],[0],.....],
                          [[1],[1],[1],...],
                          [[2],[2],[2],...]]
        cat之后：
        targets.shape = [na,nt,7]
        targets = [[[image1,class1,x1,y1,w1,h1,0],
                    [image2,class2,x2,y2,w2,h2,0],
                    ...
                    [imageN,classN,xN,yN,wN,hN,0]],
                    [[image1,class1,x1,y1,w1,h1,1],
                     [image2,class2,x2,y2,w2,h2,1],
                    ...],
                    [[image1,class1,x1,y1,w1,h1,2],
                     [image2,class2,x2,y2,w2,h2,2],
                    ...]]
        这么做是为了纪录每个label对应的anchor。
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        '''
        定义每个grid偏移量，会根据标签在grid中的相对位置来进行偏移
        '''
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            '''
            原本yaml中加载的anchors.shape = [3,6],但在yolo.py的Detect中已经通过代码
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a) 
            将anchors进行了reshape。
            self.anchors.shape = [3,3,2]
            anchors.shape = [3,2]
            '''
            anchors, shape = self.anchors[i], p[i].shape
            '''
            p是一个，nl(3)个预测层输出组成的列表，每个原始的tensor.shape = [bs(batch_size),na(anchors数),w(宽),h(高),85(4(边框坐标)+1(置信度)+80(类别数))]
            shape = p[i].shape = [bs,na,nx,ny,no] = [bs(batch_size),na(anchors数),w(宽),h(高),85(4(边框坐标)+1(置信度)+80(类别数))]
            gain = [1,1,nx,ny,nx,ny,1]
            '''
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            '''
            因为targets进行了归一化，默认在w = 1, h =1 的坐标系中，
            需要将其映射到当前输出层w = nx, h = ny的坐标系中。这里第一个输出层的宽和高都是80，所以乘80
            '''
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                """
                t[..., 4:6]存放的是真实目标的w和h,将得到每个真实标签与当前每个anchor宽高上的比值，
                """
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                '''
                torch.max(r, 1. / r)求出最大的宽比和最大的长比，shape = [3,nt,2]
                再max(2)求出同一标签中宽比和长比较大的一个，shape = [2，3,nt],之所以第一个维度变成2，
                因为torch.max如果不是比较两个tensor的大小，而是比较1个tensor某一维度的大小，则会返回values和indices：
                    torch.return_types.max(
                        values=tensor([...]),
                        indices=tensor([...]))
                所以还需要加上索引0获取values，
                torch.max(r, 1. / r).max(2)[0].shape = [3,nt],
                将其和hyp.yaml中的anchor_t超参比较，小于该值则认为标签属于当前输出层的anchor
                j = [[bool,bool,....],[bool,bool,...],[bool,bool,...]]
                j.shape = [3,nt]
                j就是anchors比值小于4的
                '''
                # 找出比值r在（r,1/r）直接的anchors
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                """
                假设j中有NTrue个True值，则
                t[j].shape = [NTrue,7]
                这个过滤后就是这一次预测层的真实anchors
                """
                t = t[j]  # filter  31*7

                # Offsets
                gxy = t[:, 2:4]  # grid xy 真实anchors的中心点的坐标
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # 计算中心点距离单元格的左边界和上边界是否在0.5的范围内（中心点不在边缘单元格内左边和上边）
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # 计算中心点距离单元格的右边界和下边界是否在0.5的范围内（中心点不在边缘单元格内右边和下边）
                '''
                j.shape = [5,NTrue]
                t.repeat之后shape为[5,NTrue,7], 
                通过索引j后t.shape = [NOff,7],NOff表示NTrue + (j,k,l,m中True的总数量)
                torch.zeros_like(gxy)[None].shape = [1,NTrue,2]
                off[:, None].shape = [5,1,2]
                相加之和shape = [5,NTrue,2]
                通过索引j后offsets.shape = [NOff,2]
                这段代码的表示当标签在grid左侧半部分时，会将标签往左偏移0.5个grid，上下右同理。
                '''
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # 距离单元格左 上，右，下的边界距离
                t = t.repeat((5, 1, 1))[j]  # 按照[全部取, 左满足，上满足，右满足，下满足]的方式取出
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 按照offset按照上面偏移转换成矩阵  offset = [全部取, 左满足，上满足，右满足，下满足] + offset ，因此真实anchors的中心点位置就是该偏移加上中心点坐标
            else:
                t = targets[0]
                offsets = 0

            # Define  chunk(a,b),a表示分成的块数，b=0沿横向分割，b=1沿纵向分割
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy(真实边框的中心点), grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()  # 计算考虑偏移的情况下，真实边框对应的单元格的左上角坐标
            gi, gj = gij.T  # grid indices  x, y
            '''
            a:所有anchor的索引 shape = [NOff]
            b:标签所属image的索引 shape = [NOff]
            gj.clamp_(0, shape[2] - 1)将标签所在grid的y限定在0到h-1之间
            gi.clamp_(0, shape[3] - 1)将标签所在grid的x限定在0到w-1之间
            indices = [image, anchor, gridy, gridx] 最终shape = [nl,4,NOff]
            tbox存放的是标签在所在grid内的相对坐标，∈[0,1] 最终shape = [nl,NOff]
            anch存放的是anchors 最终shape = [nl,NOff,2]
            tcls存放的是标签的分类 最终shape = [nl,NOff]
            '''
            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 对于某个点其实这个是要存三个值
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
