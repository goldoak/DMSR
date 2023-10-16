import torch
import torch.nn as nn

from lib.pspnet import PSPNet
from lib.pointnet import Pointnet2MSG
from lib.adaptor import PriorAdaptor


class SPGANet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024, num_structure_points=128, mode='train'):
        super(SPGANet, self).__init__()
        self.n_cat = n_cat
        self.mode = mode
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18', in_dim=3)
        self.psp_depth = PSPNet(bins=(1, 2, 3, 6), backend='resnet18', in_dim=4)
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.instance_depth = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.img_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.point_correction = nn.Sequential(
            nn.Conv1d(1027, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, n_cat * 3, 1),
        )

        self.instance_geometry = Pointnet2MSG(0)
        self.num_structure_points = num_structure_points

        conv1d_stpts_prob_modules = []
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.ReLU())
        conv1d_stpts_prob_modules.append(
            nn.Conv1d(in_channels=256, out_channels=self.num_structure_points, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)

        self.lowrank_projection = None
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.category_local = Pointnet2MSG(0)

        self.prior_enricher = PriorAdaptor(emb_dims=64, n_heads=4)

        self.category_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat * nv_prior, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat * 3, 1),
        )
        self.deformation[4].weight.data.normal_(0, 0.0001)

        self.scale = nn.Sequential(
            nn.Conv1d(3072, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, n_cat, 1),
        )

    def get_prior_enricher_lowrank_projection(self):
        return self.prior_enricher.get_lowrank_projection()

    def forward(self, pred_depth, img, choose, cat_id, prior, points=None):
        bs, n_pts = choose.size()[:2]
        nv = prior.size()[1]
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat

        out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = self.instance_color(emb)
        img_global = self.img_global(emb)

        out_depth = self.psp_depth(pred_depth)
        depth_emb = out_depth.view(bs, di, -1)
        depth_emb = torch.gather(depth_emb, 2, choose).contiguous()
        depth_emb = self.instance_depth(depth_emb)

        inst_local = torch.cat((depth_emb, emb), dim=1)  # bs x 128 x n_pts
        inst_global = self.instance_global(inst_local)  # bs x 1024 x 1

        self.lowrank_projection = self.conv1d_stpts_prob(inst_local)
        if self.mode == 'train':
            weighted_xyz = torch.sum(self.lowrank_projection[:, :, :, None] * points[:, None, :, :], dim=2)
        else:
            weighted_xyz = None

        weighted_points_features = torch.sum(self.lowrank_projection[:, None, :, :] * depth_emb[:, :, None, :], dim=3)
        weighted_img_features = torch.sum(self.lowrank_projection[:, None, :, :] * emb[:, :, None, :], dim=3)

        # category-specific features
        cat_points = self.category_local(prior)  # bs x 64 x n_pts
        cat_color = self.prior_enricher(cat_points, weighted_points_features, weighted_img_features)
        cat_local = torch.cat((cat_points, cat_color), dim=1)
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1

        # assignemnt matrix
        assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)  # bs x 2176 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()  # bs, nc*nv, n_pts -> bs*nc, nv, n_pts

        assign_mat = torch.index_select(assign_mat, 0, index)  # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()  # bs x n_pts x nv

        # deformation field
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)), dim=1)  # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)
        deltas = deltas.view(-1, 3, nv).contiguous()  # bs, nc*3, nv -> bs*nc, 3, nv
        deltas = torch.index_select(deltas, 0, index)  # bs x 3 x nv
        deltas = deltas.permute(0, 2, 1).contiguous()  # bs x nv x 3

        # mean scale offset
        scale_feat = torch.cat((img_global, inst_global, cat_global), dim=1)  # bs x 3072 x 1
        scale_offset = self.scale(scale_feat)
        scale_offset = scale_offset.view(-1, 1).contiguous()  # bs, nc, 1 -> bs*nc, 1
        scale_offset = torch.index_select(scale_offset, 0, index)  # bs x 1
        scale_offset = scale_offset.contiguous()  # bs x 1

        return weighted_xyz, assign_mat, deltas, scale_offset
