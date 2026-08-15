import torch
import torch.nn as nn
from multi_small_model.ehr_module import EHR_module
from multi_small_model.cxr_module import CXR_module
from multi_small_model.note_module import Note_module


class Mul_module(nn.Module):

    def __init__(
        self,
        device='cpu',
        embed_size=512,
        mode='joint',
        modality='ecn',
        num_classes=2,
        tasks=("mortality", "longstay", "readmission"),   # ✅ 新增
    ):
        super().__init__()
        self.device = device
        self.mode = mode
        self.embed_size = embed_size
        self.modality = modality
        self.num_classes = num_classes
        self.tasks = list(tasks)

        # ---- 子模块 ----
        self.proto_proj = nn.ModuleDict()

        # EHR
        if 'e' in modality:
            self.ehr = EHR_module(device=device, embed_size=embed_size // 2)  # 256
            self.e_dim = embed_size  # 512（你原来的设定）
            self.proto_proj['e'] = nn.Linear(self.e_dim, 768)
        else:
            self.e_dim = 0

        # CXR
        if 'c' in modality:
            self.cxr = CXR_module(device=device, embed_size=embed_size)  # 512
            self.c_dim = embed_size
            self.proto_proj['c'] = nn.Linear(self.c_dim, 1536)
        else:
            self.c_dim = 0

        # NOTE
        if 'n' in modality:
            self.note = Note_module(device=device, embed_size=embed_size)  # 512
            self.n_dim = embed_size
            self.proto_proj['n'] = nn.Linear(self.n_dim, 5120)
        else:
            self.n_dim = 0


        self.in_dim = self.e_dim + self.c_dim + self.n_dim

        if self.mode == 'joint':

            self.heads = nn.ModuleDict({
                t: nn.Linear(self.in_dim, self.num_classes) for t in self.tasks
            })

        elif self.mode == 'late':

            if 'e' in modality:
                self.cls_head_ehr = nn.Linear(self.e_dim, self.num_classes)
            if 'c' in modality:
                self.cls_head_cxr = nn.Linear(self.c_dim, self.num_classes)
            if 'n' in modality:
                self.cls_head_note = nn.Linear(self.n_dim, self.num_classes)

        else:

            self.cls_head = nn.Sequential(
                nn.Linear(self.in_dim, embed_size),
                nn.Linear(embed_size, self.num_classes)
            )

    def _zero_feat(self, B, dim, device, dtype=torch.float32):
        return torch.zeros((B, dim), device=device, dtype=dtype)

    @torch.no_grad()
    def _check_dim(self, x, expect, name):
        if x.shape[1] != expect:
            raise RuntimeError(
                f"[DimMismatch] {name} dim={x.shape[1]} != expect {expect}. "
                f"请检查对应 module.embedding 输出维度，或修改 e_dim/c_dim/n_dim 设置。"
            )

    def _encode_masked(self, mask, demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes):

        device = demo.device
        B = demo.shape[0]
        feats = []

        # ehr
        if 'e' in self.modality:
            if mask.get("ehr", False):
                e = self.ehr.embedding(demo, chart=chart, lab=lab, procedure=procedure, time=time, pooling=True)
            else:
                e = self._zero_feat(B, self.e_dim, device=device)
            feats.append(e)

        # cxr
        if 'c' in self.modality:
            if mask.get("cxr", False):
                c = self.cxr.embedding(imgs, view_points, timestamps, pooling=True)
            else:
                c = self._zero_feat(B, self.c_dim, device=device)
            feats.append(c)

        # note
        if 'n' in self.modality:
            if mask.get("note", False):
                n = self.note.embedding(notes)
            else:
                n = self._zero_feat(B, self.n_dim, device=device)
            feats.append(n)

        feat = torch.cat(feats, dim=1)  # (B, in_dim)
        return feat

    def forward_masked(self, mask, demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes,
                       return_feat: bool = False):
        if self.mode != "joint":
            raise RuntimeError("forward_masked 当前只支持 mode='joint'。")

        feat = self._encode_masked(mask, demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes)
        out = {t: self.heads[t](feat) for t in self.tasks}
        if return_feat:
            return out, feat
        return out

    def forward(
        self,
        demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes,
        ehr_embed=None, cxr_embed=None, note_embed=None,
        return_feat: bool = False
    ):

        if self.mode == 'joint':
            # 全模态 forward（不处理缺失，缺失请用 forward_masked）
            feats = []
            if 'e' in self.modality:
                e = self.ehr.embedding(demo, chart=chart, lab=lab, procedure=procedure, time=time, pooling=True)
                feats.append(e)
            if 'c' in self.modality:
                c = self.cxr.embedding(imgs, view_points, timestamps, pooling=True)
                feats.append(c)
            if 'n' in self.modality:
                n = self.note.embedding(notes)
                feats.append(n)

            feat = torch.cat(feats, dim=1)
            pred = {t: self.heads[t](feat) for t in self.tasks}

        elif self.mode == 'early':
            feats = []
            if 'e' in self.modality: feats.append(ehr_embed)
            if 'c' in self.modality: feats.append(cxr_embed)
            if 'n' in self.modality: feats.append(note_embed)
            feat = torch.cat(feats, dim=1)
            pred = self.cls_head(feat)

        elif self.mode == 'late':
            preds = []
            if 'e' in self.modality:
                e = self.ehr.embedding(demo, chart=chart, lab=lab, procedure=procedure, time=time, pooling=True)
                preds.append(self.cls_head_ehr(e))
            if 'c' in self.modality:
                c = self.cxr.embedding(imgs, view_points, timestamps, pooling=True)
                preds.append(self.cls_head_cxr(c))
            if 'n' in self.modality:
                n = self.note.embedding(notes)
                preds.append(self.cls_head_note(n))
            pred = torch.mean(torch.stack(preds, dim=0), dim=0)
            feat = None

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if return_feat:
            return pred, feat
        return pred
