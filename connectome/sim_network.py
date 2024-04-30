from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models
import torchvision.transforms.functional


@dataclass(frozen=True)
class SimilarityNetOutput:
    feature: torch.Tensor
    theta  : torch.Tensor
    side   : torch.Tensor

    @cached_property
    def length(self):
        return len(self)

    @cached_property
    def feature_np(self) -> np.ndarray:
        return self.feature.detach().cpu().numpy()
    @cached_property
    def theta_np(self) -> np.ndarray:
        return self.theta.detach().cpu().numpy()
    @cached_property
    def side_np(self) -> np.ndarray:
        return self.side.detach().cpu().numpy()


    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, index):
        return SimilarityNetOutput(self.feature[index], self.theta[index], self.side[index])

    @staticmethod
    def merge_batch(*outputs: 'SimilarityNetOutput'):
        return SimilarityNetOutput(
            torch.cat([o.feature for o in outputs], dim=0),
            torch.cat([o.theta for o in outputs], dim=0),
            torch.cat([o.side for o in outputs], dim=0),
        )

class SimilarityNet(torch.nn.Module):
    def __init__(self, base_model, output_dim=128, dropout=0.5):
        super().__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.base_model.fc.in_features, output_dim)
        self.fc_after = torch.nn.Identity()
        self.fc_after = torch.nn.Tanh()

        self.fc_theta = torch.nn.Linear(self.base_model.fc.in_features, 1)
        self.fc_theta_act = torch.nn.Tanh()

        self.fc_side = torch.nn.Linear(self.base_model.fc.in_features, 2) # 0 朝左，1 朝右
        self.fc_side_act = torch.nn.Softmax(dim=1)

        # self.fc_after = torch.nn.BatchNorm1d(output_dim)
        # self.fc_after = torch.nn.LogSigmoid()
        # self.act = torch.nn.ReLU()
        # self.act = torch.nn.Softmax()

        self.base_model.fc = torch.nn.Identity()

        self.output_dim = output_dim
        self.dropout_ratio = dropout

    def forward(self, image1, image2):
        x1 = self.base_model(image1)
        x2 = self.base_model(image2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1_theta = (self.fc_theta_act(self.fc_theta(x1))) # -1 ~ 1, 乘 180 换算到角度
        x2_theta = (self.fc_theta_act(self.fc_theta(x2)))

        # x1_side = self.fc_side_act(self.fc_side(x1))
        # x2_side = self.fc_side_act(self.fc_side(x2))
        x1_side = (self.fc_side(x1))
        x2_side = (self.fc_side(x2))
        x1 = self.fc(x1)
        x2 = self.fc(x2)


        # print(x1, self.fc_after(x1))
        x1 = self.fc_after(x1)
        x2 = self.fc_after(x2)

        # x1 = self.act(x1)
        # x2 = self.act(x2)
        return x1, x2, x1_theta, x2_theta, x1_side, x2_side

    def calc_feature(self, image):
        assert len(image.shape) == 4, "image shape should be (batch, channel, height, width)"
        
        B      = image.shape[0]
        image1 = image[:B//2]
        image2 = image[B//2:]

        x1, x2, x1_theta, x2_theta, x1_side, x2_side = self.forward(image1, image2)

        x     = torch.cat([x1, x2], dim=0)
        theta = torch.cat([x1_theta, x2_theta], dim=0)
        side  = torch.cat([x1_side, x2_side], dim=0)

        return SimilarityNetOutput(x, theta, side)

    def save_model(self, log_dir: str | Path, /, *, name: str = 'model-final.pth'):
        torch.save(self.state_dict(), Path(log_dir) / name)

    def load_model(self, log_dir: str | Path, /, *, name: str = 'model-final.pth'):
        self.load_state_dict(torch.load(Path(log_dir) / name))
