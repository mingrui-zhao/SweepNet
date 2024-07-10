# Code adopted from the official POCO repository: https://github.com/valeoai/POCO
import torch
import torch.nn as nn
import torch_geometric
import logging
import time
from lightconvpoint.nn import Convolution_FKAConv as Conv
from lightconvpoint.nn import max_pool, interpolate
from lightconvpoint.spatial import knn, sampling_quantized as sampling
from lightconvpoint.utils.functional import batch_gather
from torch_geometric.data import Data
import yaml

NormLayer = nn.BatchNorm1d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Network(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, decoder_kwargs, **kwargs):
        super().__init__()

        self.net = FKAConvNetwork(in_channels, latent_size, segmentation=True, dropout=0)
        self.projection = InterpAttentionKHeadsNet(latent_size, out_channels, decoder_kwargs)
        self.lcp_preprocess = True

        logging.info(f"Network -- backbone -- {count_parameters(self.net)} parameters")
        logging.info(f"Network -- projection -- {count_parameters(self.projection)} parameters")

    def forward(self, data, spatial_only=False, spectral_only=False):

        if spatial_only:
            net_data = self.net(data, spatial_only=spatial_only)
            if "output_support" in net_data:
                data["output_support"] = net_data["output_support"]
            proj_data = self.projection.forward_spatial(data)
            net_data["proj_indices"] = proj_data["proj_indices"]
            return net_data

        if not spectral_only:
            spatial_data = self.net.forward_spatial(data)
            if "output_support" in spatial_data:
                data["output_support"] = spatial_data["output_support"]
            proj_data = self.projection.forward_spatial(data)
            spatial_data["proj_indices"] = proj_data["proj_indices"]
            for key, value in spatial_data.items():
                data[key] = value

        latents = self.net(data, spectral_only=False)
        data["latents"] = latents
        ret_data = self.projection(data, spectral_only=True)

        return ret_data



    def get_latent(self, data, with_correction=False, spatial_only=False, spectral_only=False):

        latents = self.net(data, spatial_only=spatial_only, spectral_only=spectral_only)
        data["latents"] = latents

        data["proj_correction"] = None
        if with_correction:
            data_in_proj = {"latents":latents, "pos":data["pos"], "pos_non_manifold":data["pos"].clone(), "proj_correction":None}
            data_proj = self.projection(data_in_proj, spectral_only=False)
            data["proj_correction"] = data_proj
        return data

    def from_latent(self, data):
        data_proj = self.projection(data)
        return data_proj#["outputs"]
    
class InterpAttentionKHeadsNet(torch.nn.Module):

    def __init__(self, latent_size, out_channels, K=16):
        super().__init__()
        
        logging.info(f"InterpNet - Simple - K={K}")
        # self.projection_layer = FKAConv(latent_size, latent_size, 16, sampling=None, neighborhood_search=knn, neighborhood_size=16, ratio=1)
        self.fc1 = torch.nn.Conv2d(latent_size+3, latent_size, 1)
        self.fc2 = torch.nn.Conv2d(latent_size, latent_size, 1)
        self.fc3 = torch.nn.Conv2d(latent_size, latent_size, 1)

        self.fc8 = torch.nn.Conv1d(latent_size, out_channels, 1)
        self.activation = torch.nn.ReLU()

        self.fc_query = torch.nn.Conv2d(latent_size, 64, 1)
        self.fc_value = torch.nn.Conv2d(latent_size, latent_size,1)

        self.k = K

    def forward_spatial(self, data):

        pos = data["pos"]
        pos_non_manifold = data["pos_non_manifold"]

        add_batch_dimension_pos = False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            add_batch_dimension_pos = True

        add_batch_dimension_non_manifold = False
        if len(pos_non_manifold.shape) == 2:
            pos_non_manifold = pos_non_manifold.unsqueeze(0)
            add_batch_dimension_non_manifold = True

        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)

        indices = knn(pos, pos_non_manifold, self.k)

        if add_batch_dimension_non_manifold or add_batch_dimension_pos:
            indices = indices.squeeze(0)

        ret_data = {}
        ret_data["proj_indices"] = indices

        return ret_data


    def forward(self, data, spatial_only=False, spectral_only=False, last_layer=True, return_last_features=False):
        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:
            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value

        
        x = data["latents"]
        indices = data["proj_indices"]
        pos = data["pos"]
        pos_non_manifold = data["pos_non_manifold"]


        if pos.shape[1] != 3:
            pos = pos.transpose(1,2)

        if pos_non_manifold.shape[1] != 3:
            pos_non_manifold = pos_non_manifold.transpose(1,2)

        x = batch_gather(x, 2, indices)
        pos = batch_gather(pos, 2, indices)
        pos = pos_non_manifold.unsqueeze(3) - pos

        x = torch.cat([x,pos], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        query = self.fc_query(x)
        value = self.fc_value(x)

        attention = torch.nn.functional.softmax(query, dim=-1).mean(dim=1)
        x = torch.matmul(attention.unsqueeze(-2), value.permute(0,2,3,1)).squeeze(-2)
        x = x.transpose(1,2)
        
        if return_last_features:
            xout = self.fc8(x)
            return xout, x

        if last_layer:
            x = self.fc8(x)

        return x
    
class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,adaptive_normalization=True):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels//2, 1)
        self.bn0 = NormLayer(in_channels//2)
        self.cv1 = Conv(in_channels//2, in_channels//2, kernel_size, adaptive_normalization=adaptive_normalization)
        self.bn1 = NormLayer(in_channels//2)
        self.cv2 = nn.Conv1d(in_channels//2, out_channels, 1)
        self.bn2 = NormLayer(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = NormLayer(out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, pos, support_points, neighbors_indices):

        x_short = x
        x = self.activation(self.bn0(self.cv0(x)))
        x = self.activation(self.bn1(self.cv1(x, pos, support_points, neighbors_indices)))
        x = self.bn2(self.cv2(x))

        x_short = self.bn_shortcut(self.shortcut(x_short))
        if x_short.shape[2] != x.shape[2]:
            x_short = max_pool(x_short, neighbors_indices)

        x = self.activation(x + x_short)

        return x


class FKAConvNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, dropout=0.5, last_layer_additional_size=None, adaptive_normalization=True, fix_support_number=False):
        super().__init__()

        self.lcp_preprocess = True
        self.segmentation = segmentation
        self.adaptive_normalization = adaptive_normalization
        self.fix_support_point_number = fix_support_number

        self.cv0 = Conv(in_channels, hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.bn0 = NormLayer(hidden)

        
        self.resnetb01 = ResidualBlock(hidden, hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb10 = ResidualBlock(hidden, 2*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb11 = ResidualBlock(2*hidden, 2*hidden, 16, adaptive_normalization=self.adaptive_normalization) 
        self.resnetb20 = ResidualBlock(2*hidden, 4*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb21 = ResidualBlock(4*hidden, 4*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb30 = ResidualBlock(4*hidden, 8*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb31 = ResidualBlock(8*hidden, 8*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb40 = ResidualBlock(8*hidden, 16*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, 16, adaptive_normalization=self.adaptive_normalization)
        if self.segmentation:

            self.cv5 = nn.Conv1d(32*hidden, 16 * hidden, 1)
            self.bn5 = NormLayer(16*hidden)
            self.cv3d = nn.Conv1d(24*hidden, 8 * hidden, 1)
            self.bn3d = NormLayer(8 * hidden)
            self.cv2d = nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = NormLayer(4 * hidden)
            self.cv1d = nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = NormLayer(2 * hidden)
            self.cv0d = nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = NormLayer(hidden)

            if last_layer_additional_size is not None:
                self.fcout = nn.Conv1d(hidden+last_layer_additional_size, out_channels, 1)
            else:
                self.fcout = nn.Conv1d(hidden, out_channels, 1)
        else:
            self.fcout = nn.Conv1d(16*hidden, out_channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()


    def forward_spatial(self, data):

        pos = data["pos"].clone()

        add_batch_dimension = False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            add_batch_dimension = True
        
        # compute the support points
        if self.fix_support_point_number:
            support1, _ = sampling(pos, n_support=512)
            support2, _ = sampling(support1, n_support=128)
            support3, _ = sampling(support2, n_support=32)
            support4, _ = sampling(support3, n_support=8)
        else:
            support1, _ = sampling(pos, 0.25)
            support2, _ = sampling(support1, 0.25)
            support3, _ = sampling(support2, 0.25)
            support4, _ = sampling(support3, 0.25)


        # compute the ids
        ids00 = knn(pos, pos, 16)
        ids01 = knn(pos, support1, 16)
        ids11 = knn(support1, support1, 16)
        ids12 = knn(support1, support2, 16)
        ids22 = knn(support2, support2, 16)
        ids23 = knn(support2, support3, 16)
        ids33 = knn(support3, support3, 16)
        ids34 = knn(support3, support4, 16)
        ids44 = knn(support4, support4, 16)
        if self.segmentation:
            ids43 = knn(support4, support3, 1)
            ids32 = knn(support3, support2, 1)
            ids21 = knn(support2, support1, 1)
            ids10 = knn(support1, pos, 1)

        ret_data = {}
        if add_batch_dimension:
            support1 = support1.squeeze(0)
            support2 = support2.squeeze(0)
            support3 = support3.squeeze(0)
            support4 = support4.squeeze(0)
            ids00 = ids00.squeeze(0)
            ids01 = ids01.squeeze(0)
            ids11 = ids11.squeeze(0)
            ids12 = ids12.squeeze(0)
            ids22 = ids22.squeeze(0)
            ids23 = ids23.squeeze(0)
            ids33 = ids33.squeeze(0)
            ids34 = ids34.squeeze(0)
            ids44 = ids44.squeeze(0)

        ret_data["support1"] = support1
        ret_data["support2"] = support2
        ret_data["support3"] = support3
        ret_data["support4"] = support4

        ret_data["ids00"] = ids00
        ret_data["ids01"] = ids01
        ret_data["ids11"] = ids11
        ret_data["ids12"] = ids12
        ret_data["ids22"] = ids22
        ret_data["ids23"] = ids23
        ret_data["ids33"] = ids33
        ret_data["ids34"] = ids34
        ret_data["ids44"] = ids44

        if self.segmentation:
            if add_batch_dimension:
                ids43 = ids43.squeeze(0)
                ids32 = ids32.squeeze(0)
                ids21 = ids21.squeeze(0)
                ids10 = ids10.squeeze(0)
            
            ret_data["ids43"] = ids43
            ret_data["ids32"] = ids32
            ret_data["ids21"] = ids21
            ret_data["ids10"] = ids10
        
        
        return ret_data


    def forward(self, data, spatial_only=False, spectral_only=False, cat_in_last_layer=None):


        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:

            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value
            # data = {**data, **spatial_data}

        x = data["x"]
        pos = data["pos"]

        x0 = self.activation(self.bn0(self.cv0(x, pos, pos, data["ids00"])))
        x0 = self.resnetb01(x0, pos, pos, data["ids00"])
        x1 = self.resnetb10(x0, pos, data["support1"], data["ids01"])
        x1 = self.resnetb11(x1, data["support1"], data["support1"], data["ids11"])
        x2 = self.resnetb20(x1, data["support1"], data["support2"], data["ids12"])
        x2 = self.resnetb21(x2, data["support2"], data["support2"], data["ids22"])
        x3 = self.resnetb30(x2, data["support2"], data["support3"], data["ids23"])
        x3 = self.resnetb31(x3, data["support3"], data["support3"], data["ids33"])
        x4 = self.resnetb40(x3, data["support3"], data["support4"], data["ids34"])
        x4 = self.resnetb41(x4, data["support4"], data["support4"], data["ids44"])

        if self.segmentation:
            
            x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
            x4d = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
            x4d = x4

            x3d = interpolate(x4d, data["ids43"])
            x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

            x2d = interpolate(x3d, data["ids32"])
            x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))
            
            x1d = interpolate(x2d, data["ids21"])
            x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))
            
            xout = interpolate(x1d, data["ids10"])
            xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
            xout = self.dropout(xout)
            if cat_in_last_layer is not None:
                xout = torch.cat([xout, cat_in_last_layer.expand(-1,-1,xout.shape[2])], dim=1)
            xout = self.fcout(xout)

        else:

            xout = x4
            xout = self.dropout(xout)
            xout = self.fcout(xout)
            xout = xout.mean(dim=2)

        return xout
    
def get_poco_model(config):
    config = yaml.safe_load(open(config, "r"))
    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    decoder_kwargs = config['network_decoder_k']
    def network_function():
        return Network(3, latent_size, N_LABELS, decoder_kwargs)
    net = network_function()
    net.cuda()
    return net