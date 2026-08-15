import torch
import torch.nn as nn
from multi_small_model.ehr_module import EHR_module
from multi_small_model.cxr_module import CXR_module
from multi_small_model.note_module import Note_module
import numpy as np

class ClientMulStudent(nn.Module):
    def __init__(self, device='cpu', embed_size=512, mode='joint', modality='ecn', dropout=0.0):
        super().__init__()
        self.device = device
        self.mode = mode
        self.embed_size = embed_size
        self.modality = modality


        if 'e' in modality:

            self.ehr = EHR_module(device=device, embed_size=embed_size // 2)
        else:
            self.ehr = None

        if 'c' in modality:
            self.cxr = CXR_module(device=device, embed_size=embed_size)
        else:
            self.cxr = None

        if 'n' in modality:
            self.note = Note_module(device=device, embed_size=embed_size)
        else:
            self.note = None


        feat_dim = self._infer_feat_dim()


        if mode == 'joint':
            self.cls_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 2),
            )
        elif mode == 'late':

            raise NotImplementedError("建议先用 mode='joint' 跑通 KD/CE，再考虑 late。")
        elif mode == 'early':
            raise NotImplementedError("early 需要外部传各模态 embed，不建议现在用。")
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(feat_dim, embed_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_size, 2),
            )

    @torch.no_grad()
    def _infer_feat_dim(self):

        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")


        demo = torch.zeros(1, 8, device=dev)
        chart = [[]]
        lab = [[]]
        procedure = [[]]
        time = [np.zeros((0,), dtype=np.float32)]

        imgs = [[torch.zeros(1, 128, 128)]]
        view_points = [[0]]
        timestamps = [[0.0]]

        notes = [""]

        embeds = []
        if self.ehr is not None:
            e = self.ehr.embedding(demo, chart, lab, procedure, time, pooling=True)
            embeds.append(e)
        if self.cxr is not None:
            c = self.cxr.embedding(imgs, view_points, timestamps, pooling=True)
            embeds.append(c)
        if self.note is not None:
            n = self.note.embedding(notes)
            embeds.append(n)

        feat = torch.cat(embeds, dim=1)
        return int(feat.size(1))

    def embedding(self, ehr_pack, cxr_pack, notes):

        embeds = []

        if 'e' in self.modality:
            assert ehr_pack is not None
            demo, chart, lab, procedure, time = ehr_pack
            embeds.append(self.ehr.embedding(demo, chart, lab, procedure, time, pooling=True))

        if 'c' in self.modality:
            assert cxr_pack is not None
            imgs, view_points, timestamps = cxr_pack
            embeds.append(self.cxr.embedding(imgs, view_points, timestamps, pooling=True))

        if 'n' in self.modality:
            assert notes is not None
            embeds.append(self.note.embedding(notes))

        feat = torch.cat(embeds, dim=1)
        return feat

    def forward(self, ehr_pack, cxr_pack, notes, return_feat=False):

        feat = self.embedding(ehr_pack, cxr_pack, notes)
        logits = self.cls_head(feat)
        if return_feat:
            return logits, feat
        return logits
