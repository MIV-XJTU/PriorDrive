'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from .xbert import BertConfig, BertModel, VecPointEmbeddings
import torch
from torch import nn, Tensor
import random
import numpy as np


class ContextEncodrNet(nn.Module):
    def __init__(self, type):
        super().__init__()
        embed_dim = 64     
        self.point_embedding = VecPointEmbeddings(embed_dim)

        config = './projects/mmdet3d_plugin/UVE/config_bert_context.json'
        bert_config = BertConfig.from_json_file(config)
        self.vec_encoder = BertModel(bert_config, type, add_pooling_layer=False, sign=False)

        self.upsampling = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            )
        


    def forward(self, points, attention_mask):
        final_vec_point_embedding = self.point_embedding._embed_points(points, attention_mask)


        vec_output = self.vec_encoder(attention_mask=attention_mask, encoder_embeds=final_vec_point_embedding,
                                             return_dict=True, mode='multi_modal')

        vec_embeds = vec_output.last_hidden_state

        return self.upsampling(vec_embeds)

    

