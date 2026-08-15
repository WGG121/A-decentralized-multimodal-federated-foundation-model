import torch
import torch.nn as nn

from multi_small_model.ehr_module import EHR_module
from multi_small_model.cxr_module import CXR_module
from multi_small_model.note_module import Note_module


class Mul_module(nn.Module):


    def __init__(self, device='cpu', embed_size=512, mode='joint', modality='ecn'):
        super().__init__()
        self.device = device
        self.mode = mode
        self.embed_size = embed_size
        self.modality = modality

        self.e_dim = embed_size // 2
        self.c_dim = embed_size
        self.n_dim = embed_size


        if 'e' in modality:
            self.ehr = EHR_module(device=device, embed_size=self.e_dim)
        if 'c' in modality:
            self.cxr = CXR_module(device=device, embed_size=self.c_dim)
        if 'n' in modality:
            self.note = Note_module(device=device, embed_size=self.n_dim)


        feat_dim = 0
        if 'e' in modality:
            feat_dim += self.e_dim
        if 'c' in modality:
            feat_dim += self.c_dim
        if 'n' in modality:
            feat_dim += self.n_dim
        self.feat_dim = feat_dim


        if mode == 'joint':
            self.cls_head = nn.Linear(self.feat_dim, 2)
        elif mode == 'late':
            if 'e' in modality:
                self.cls_head_ehr = nn.Linear(self.e_dim, 2)
            if 'c' in modality:
                self.cls_head_cxr = nn.Linear(self.c_dim, 2)
            if 'n' in modality:
                self.cls_head_note = nn.Linear(self.n_dim, 2)
        elif mode == 'early':

            self.cls_head = nn.Linear(self.feat_dim, 2)
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(self.feat_dim, embed_size),
                nn.ReLU(inplace=True),
                nn.Linear(embed_size, 2)
            )

    @torch.no_grad()
    def _check_none(self, x, name):
        if x is None:
            raise ValueError(f"[Mul_module] {name} is None but required by modality='{self.modality}'")

    def embedding(self, demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes):

        embeds = []

        if 'e' in self.modality:
            # 如果你数据端会给缺失ehr=None，那就这里检查或改成允许None+mask
            self._check_none(demo, "demo")
            ehr_embeds = self.ehr.embedding(demo, chart, lab, procedure, time, pooling=True)  # (B, e_dim)
            embeds.append(ehr_embeds)

        if 'c' in self.modality:
            self._check_none(imgs, "imgs")
            cxr_embeds = self.cxr.embedding(imgs, view_points, timestamps, pooling=True)     # (B, c_dim)
            embeds.append(cxr_embeds)

        if 'n' in self.modality:
            self._check_none(notes, "notes")
            note_embeds = self.note.embedding(notes)                                        # (B, n_dim)
            embeds.append(note_embeds)

        feat = torch.cat(embeds, dim=1)  # (B, feat_dim)
        return feat

    def forward(
        self,
        demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes,
        ehr_embed=None, cxr_embed=None, note_embed=None,
        return_feat: bool = True,
    ):

        if self.mode == 'joint':
            feat = self.embedding(demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes)
            pred = self.cls_head(feat)

        elif self.mode == 'early':

            feats = []
            if 'e' in self.modality:
                self._check_none(ehr_embed, "ehr_embed")
                feats.append(ehr_embed)
            if 'c' in self.modality:
                self._check_none(cxr_embed, "cxr_embed")
                feats.append(cxr_embed)
            if 'n' in self.modality:
                self._check_none(note_embed, "note_embed")
                feats.append(note_embed)

            feat = torch.cat(feats, dim=1)
            pred = self.cls_head(feat)

        elif self.mode == 'late':

            preds = []
            feat_chunks = []

            if 'e' in self.modality:
                self._check_none(demo, "demo")
                e = self.ehr.embedding(demo, chart, lab, procedure, time, pooling=True)
                preds.append(self.cls_head_ehr(e))
                feat_chunks.append(e)

            if 'c' in self.modality:
                self._check_none(imgs, "imgs")
                c = self.cxr.embedding(imgs, view_points, timestamps, pooling=True)
                preds.append(self.cls_head_cxr(c))
                feat_chunks.append(c)

            if 'n' in self.modality:
                self._check_none(notes, "notes")
                n = self.note.embedding(notes)
                preds.append(self.cls_head_note(n))
                feat_chunks.append(n)

            pred = torch.mean(torch.stack(preds, dim=0), dim=0)
            feat = torch.cat(feat_chunks, dim=1)

        else:
            feat = self.embedding(demo, chart, lab, procedure, time, imgs, view_points, timestamps, notes)
            pred = self.cls_head(feat)

        if return_feat:
            return pred, feat
        return pred


def load_pretrained_partial(model: nn.Module, ckpt_path: str, allow_prefixes=("ehr.", "cxr.", "note.")):

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    filtered = {k: v for k, v in sd.items() if k.startswith(allow_prefixes)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[load_pretrained_partial] loaded={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected
