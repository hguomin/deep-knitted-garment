# Guomin @2022/10/01

"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn

from utils.mesh_sampler import MeshSampler

from .graph_layers import GraphResBlock, GraphLinear
from .resnet import resnet50

class ClothGraphCNN(nn.Module):
    
    def __init__(self, A, ref_vertices, gar_mesh_sampler: MeshSampler, num_layers=5, num_channels=512):
        super(ClothGraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices

        self.resnet = resnet50(pretrained=True)

        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.gc = nn.Sequential(*layers)

        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

        # For garment encoder
        self.gar_ref_vertices = gar_mesh_sampler.ref_vertices.t()
        self.gar_A = gar_mesh_sampler.adjmat
        gar_layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        gar_layers.append(GraphResBlock(2 * num_channels, num_channels, self.gar_A))
        for i in range(num_layers):
            gar_layers.append(GraphResBlock(num_channels, num_channels, self.gar_A))
        self.gar_gc = nn.Sequential(*gar_layers)

        self.gar_shape = nn.Sequential(GraphResBlock(num_channels, 64, self.gar_A),
                                   GraphResBlock(64, 32, self.gar_A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))


        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                      nn.ReLU(inplace=True),
                                      GraphLinear(num_channels, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(A.shape[0], 3))

    def forward(self, image):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        image_resnet = self.resnet(image)
        image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        camera = self.camera_fc(x).view(batch_size, 3)

        gar_ref_vertices = self.gar_ref_vertices.expand(batch_size, -1, -1)
        gar_image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, gar_ref_vertices.shape[-1])
        y = torch.cat([gar_ref_vertices, gar_image_enc], dim=1)
        y = self.gar_gc(y)
        gar_shape = self.gar_shape(y)

        return shape, camera, gar_shape

# Guomin: Use MLP as decoder to predict both human and garment shape
class ClothGraphCNN2(nn.Module):
    def __init__(self, A, ref_vertices, gar_mesh_sampler: MeshSampler, num_layers=5, num_channels=512):
        super(ClothGraphCNN2, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices

        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.gc = nn.Sequential(*layers)

        self.shape = nn.Sequential(GraphLinear(num_channels, 64),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(64, 32),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3),
                                   nn.Identity())
        # For garment encoder
        self.gar_ref_vertices = gar_mesh_sampler.ref_vertices.t()
        self.gar_A = gar_mesh_sampler.adjmat
        gar_layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        gar_layers.append(GraphResBlock(2 * num_channels, num_channels, self.gar_A))
        for i in range(num_layers):
            gar_layers.append(GraphResBlock(num_channels, num_channels, self.gar_A))
        self.gar_gc = nn.Sequential(*gar_layers)

        self.gar_shape = nn.Sequential(GraphResBlock(num_channels, 64, self.gar_A),
                                   GraphResBlock(64, 32, self.gar_A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                      nn.ReLU(inplace=True),
                                      GraphLinear(num_channels, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(A.shape[0], 3))

    def forward(self, image_resnet):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image_resnet.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        camera = self.camera_fc(x).view(batch_size, 3)
        
        
        gar_ref_vertices = self.gar_ref_vertices.expand(batch_size, -1, -1)
        gar_image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, gar_ref_vertices.shape[-1])
        y = torch.cat([gar_ref_vertices, gar_image_enc], dim=1)
        y = self.gar_gc(y)
        gar_shape = self.gar_shape(y)
        
        return shape, camera, gar_shape

class ClothGraphCNN3(nn.Module):
    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
        super(ClothGraphCNN3, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices

        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.gc = nn.Sequential(*layers)

        self.shape = nn.Sequential(GraphLinear(num_channels, 64),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(64, 32),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3),
                                   nn.Identity())
        '''
        # For garment encoder
        self.gar_ref_vertices = gar_mesh_sampler.ref_vertices.t()
        self.gar_A = gar_mesh_sampler.adjmat
        gar_layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        gar_layers.append(GraphResBlock(2 * num_channels, num_channels, self.gar_A))
        for i in range(num_layers):
            gar_layers.append(GraphResBlock(num_channels, num_channels, self.gar_A))
        self.gar_gc = nn.Sequential(*gar_layers)

        self.gar_shape = nn.Sequential(GraphResBlock(num_channels, 64, self.gar_A),
                                   GraphResBlock(64, 32, self.gar_A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        '''

        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                      nn.ReLU(inplace=True),
                                      GraphLinear(num_channels, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(A.shape[0], 3))

    def forward(self, image_resnet):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image_resnet.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        camera = self.camera_fc(x).view(batch_size, 3)
        
        '''
        gar_ref_vertices = self.gar_ref_vertices.expand(batch_size, -1, -1)
        gar_image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, gar_ref_vertices.shape[-1])
        y = torch.cat([gar_ref_vertices, gar_image_enc], dim=1)
        y = self.gar_gc(y)
        gar_shape = self.gar_shape(y)
        '''
        return shape, camera

class ClothGraphConvNetwork(nn.Module):
    
    def __init__(self, gar_mesh_sampler: MeshSampler, num_layers=5, num_channels=512):
        super(ClothGraphConvNetwork, self).__init__()

        # cloth encoder
        self.gar_ref_vertices = gar_mesh_sampler.ref_vertices.t()
        self.gar_A = gar_mesh_sampler.adjmat
        gar_layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        gar_layers.append(GraphResBlock(2 * num_channels, num_channels, self.gar_A))
        for i in range(num_layers):
            gar_layers.append(GraphResBlock(num_channels, num_channels, self.gar_A))
        self.gar_gc = nn.Sequential(*gar_layers)

        # cloth decoder
        self.gar_shape = nn.Sequential(GraphResBlock(num_channels, 64, self.gar_A),
                                   GraphResBlock(64, 32, self.gar_A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

    def forward(self, image_resnet):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image_resnet.shape[0]

        gar_ref_vertices = self.gar_ref_vertices.expand(batch_size, -1, -1)
        gar_image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, gar_ref_vertices.shape[-1])
        y = torch.cat([gar_ref_vertices, gar_image_enc], dim=1)
        y = self.gar_gc(y)
        gar_shape = self.gar_shape(y)

        return gar_shape


class ClothGraphConvNetwork_MLPDecoder(nn.Module):
    
    def __init__(self, gar_mesh_sampler: MeshSampler, num_layers=5, num_channels=512):
        super(ClothGraphConvNetwork_MLPDecoder, self).__init__()

        # cloth encoder
        self.gar_ref_vertices = gar_mesh_sampler.ref_vertices.t()
        self.gar_A = gar_mesh_sampler.adjmat
        gar_layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        gar_layers.append(GraphResBlock(2 * num_channels, num_channels, self.gar_A))
        for i in range(num_layers):
            gar_layers.append(GraphResBlock(num_channels, num_channels, self.gar_A))
        self.gar_gc = nn.Sequential(*gar_layers)

        # cloth decoder
        #self.gar_shape = nn.Sequential(GraphResBlock(num_channels, 64, self.gar_A),
        #                           GraphResBlock(64, 32, self.gar_A),
        #                           nn.GroupNorm(32 // 8, 32),
        #                           nn.ReLU(inplace=True),
        #                           GraphLinear(32, 3))

        self.gar_shape = nn.Sequential(GraphLinear(num_channels, 64),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(64, 32),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3),
                                   nn.Identity())

    def forward(self, image_resnet):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image_resnet.shape[0]

        gar_ref_vertices = self.gar_ref_vertices.expand(batch_size, -1, -1)
        gar_image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, gar_ref_vertices.shape[-1])
        y = torch.cat([gar_ref_vertices, gar_image_enc], dim=1)
        y = self.gar_gc(y)
        gar_shape = self.gar_shape(y)

        return gar_shape

class ClothGraphConvNetwork_MLPDecoder_Fusion(nn.Module):
    
    def __init__(self, body_adjmat, gar_mesh_sampler: MeshSampler, num_layers=5, num_channels=512):
        super(ClothGraphConvNetwork_MLPDecoder_Fusion, self).__init__()

        # body encoder
        body_layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        body_layers.append(GraphResBlock(2 * num_channels, num_channels, body_adjmat))
        for i in range(num_layers):
            body_layers.append(GraphResBlock(num_channels, num_channels, body_adjmat))
        self.body_encoder = nn.Sequential(*body_layers)

        # cloth encoder
        self.gar_ref_vertices = gar_mesh_sampler.ref_vertices.t()
        self.gar_A = gar_mesh_sampler.adjmat
        gar_layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        gar_layers.append(GraphResBlock(2 * num_channels, num_channels, self.gar_A))
        for i in range(num_layers):
            gar_layers.append(GraphResBlock(num_channels, num_channels, self.gar_A))
        self.gar_gc = nn.Sequential(*gar_layers)

        # cloth decoder
        #self.gar_shape = nn.Sequential(GraphResBlock(num_channels, 64, self.gar_A),
        #                           GraphResBlock(64, 32, self.gar_A),
        #                           nn.GroupNorm(32 // 8, 32),
        #                           nn.ReLU(inplace=True),
        #                           GraphLinear(32, 3))

        self.gar_shape = nn.Sequential(GraphLinear(num_channels, 64),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(64, 32),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3),
                                   nn.Identity())

    def forward(self, image_resnet, body_vertices_sub=None, body_adjmat=None):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image_resnet.shape[0]

        body_image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, body_vertices_sub.shape[-1])
        x = torch.cat([body_vertices_sub, body_image_enc], dim=1)
        x = self.body_encoder(x)

        gar_ref_vertices = self.gar_ref_vertices.expand(batch_size, -1, -1)
        gar_image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, gar_ref_vertices.shape[-1])
        y = torch.cat([gar_ref_vertices, gar_image_enc], dim=1)
        y = self.gar_gc(y)
        gar_shape = self.gar_shape(y)

        return gar_shape