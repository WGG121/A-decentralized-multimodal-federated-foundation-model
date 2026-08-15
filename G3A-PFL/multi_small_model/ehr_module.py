import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np


def value_embedding_data(d=512, split=200):
    vec = np.array([np.arange(split) * i for i in range(d // 2)], dtype=np.float32).transpose()
    vec = vec / vec.max()
    embedding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    embedding[0, :d] = 0
    embedding = torch.from_numpy(embedding)
    return embedding


class EHR_Embedding(nn.Module):
    def __init__(self, embed_type, varible_num=1, split_num=200, embed_size=256, device='cpu'):
        super(EHR_Embedding, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.embed_type = embed_type
        if embed_type == 'demo':
            self.anchor_age = nn.Embedding(9, embed_size)
            self.insurance = nn.Embedding(3, embed_size)
            self.language = nn.Embedding(2, embed_size)
            self.marital_status = nn.Embedding(5, embed_size)
            self.ethnicity = nn.Embedding(8, embed_size)
        elif embed_type == 'chart':
            self.var = nn.Embedding(varible_num, embed_size)
            self.value = nn.Embedding.from_pretrained(value_embedding_data(embed_size, split_num))
            self.eye_opening = nn.Embedding(4, embed_size)
            self.motor_response = nn.Embedding(6, embed_size)
            self.verbal_response = nn.Embedding(6, embed_size)
            self.map = nn.Linear(2 * embed_size, embed_size)
        elif embed_type == 'lab':
            self.var = nn.Embedding(varible_num, embed_size)
            self.value = nn.Embedding.from_pretrained(value_embedding_data(embed_size, split_num))
            self.map = nn.Linear(2 * embed_size, embed_size)
        elif embed_type == 'procedure':
            self.operation = nn.Embedding(varible_num, embed_size)
        elif embed_type == 'time':
            self.value = nn.Embedding.from_pretrained(value_embedding_data(embed_size, split_num))

    def forward(self, x):


        if self.embed_type == 'demo':

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            x = x.to(self.device).long()


            anchor_age = x[:, 0]
            insurance = x[:, 1]
            language = x[:, 2]
            marital_status = x[:, 3]


            anchor_age = anchor_age - 1
            marital_status = marital_status + 1


            def clamp_index(idx, num_embeddings):
                return torch.clamp(idx, 0, num_embeddings - 1)

            anchor_age = clamp_index(anchor_age, self.anchor_age.num_embeddings)
            insurance = clamp_index(insurance, self.insurance.num_embeddings)
            language = clamp_index(language, self.language.num_embeddings)
            marital_status = clamp_index(marital_status, self.marital_status.num_embeddings)


            anchor_age_emb = self.anchor_age(anchor_age)
            insurance_emb = self.insurance(insurance)
            language_emb = self.language(language)
            marital_status_emb = self.marital_status(marital_status)


            embed = torch.stack(
                [anchor_age_emb, insurance_emb, language_emb, marital_status_emb],
                dim=1
            )

            return embed

        if self.embed_type == 'chart':

            if x is None or len(x) == 0:
                return torch.zeros(0, self.embed_size, device=self.device)

            var, value = list(zip(*x))
            var = torch.as_tensor(var, dtype=torch.long, device=self.device)
            value = torch.as_tensor(value, dtype=torch.long, device=self.device)


            var = torch.clamp(var, 0, self.var.num_embeddings - 1)

            var_embed = self.var(var)

            value_embeds = []
            for idx, v in enumerate(value):
                v_int = int(v.item())

                if var[idx].item() == 0:

                    vmax = self.eye_opening.num_embeddings - 1
                    v_int = max(0, min(v_int, vmax))
                    v_tensor = torch.tensor(v_int, dtype=torch.long, device=self.device)
                    ve = self.eye_opening(v_tensor)
                elif var[idx].item() == 1:

                    vmax = self.motor_response.num_embeddings - 1
                    v_int = max(0, min(v_int, vmax))
                    v_tensor = torch.tensor(v_int, dtype=torch.long, device=self.device)
                    ve = self.motor_response(v_tensor)
                elif var[idx].item() == 2:

                    vmax = self.verbal_response.num_embeddings - 1
                    v_int = max(0, min(v_int, vmax))
                    v_tensor = torch.tensor(v_int, dtype=torch.long, device=self.device)
                    ve = self.verbal_response(v_tensor)
                else:

                    vmax = self.value.num_embeddings - 1
                    v_int = max(0, min(v_int, vmax))
                    v_tensor = torch.tensor(v_int, dtype=torch.long, device=self.device)
                    ve = self.value(v_tensor)

                value_embeds.append(ve)

            value_embed = torch.stack(value_embeds, dim=0)

            embed = torch.cat([var_embed, value_embed], dim=1)
            embed = torch.nan_to_num(embed, nan=0.0, posinf=0.0, neginf=0.0)
            embed = self.map(embed)
            return embed

        if self.embed_type == 'lab':

            if x is None or len(x) == 0:
                return torch.zeros(0, self.embed_size, device=self.device)

            var, value = list(zip(*x))
            var = torch.as_tensor(var, dtype=torch.long, device=self.device)
            value = torch.as_tensor(value, dtype=torch.long, device=self.device)


            var = torch.clamp(var, 0, self.var.num_embeddings - 1)

            value = torch.clamp(value, 0, self.value.num_embeddings - 1)

            var_embed = self.var(var)
            value_embed = self.value(value)

            embed = torch.cat([var_embed, value_embed], dim=1)
            embed = torch.nan_to_num(embed, nan=0.0, posinf=0.0, neginf=0.0)
            embed = self.map(embed)
            return embed

        if self.embed_type == 'procedure':

            if x is None or len(x) == 0:
                return torch.zeros(0, self.embed_size, device=self.device)
            op = torch.as_tensor(x, dtype=torch.long, device=self.device)
            embed = self.operation(op)
            return embed

        if self.embed_type == 'time':

            if isinstance(x, torch.Tensor):
                value = x.to(self.device).float()
            else:

                x_np = np.array(x, dtype=np.float32)


                if x_np.size == 0:
                    x_np = np.array([0.0], dtype=np.float32)

                if x_np.ndim == 0:
                    x_np = x_np[None]

                value = torch.from_numpy(x_np).to(self.device)


            value = value / 48.0

            value = torch.div(value, (1.0 / 200.0), rounding_mode='trunc').long()
            value = torch.clamp(value, 0, 199)

            embed = self.value(value)  # [T, D]
            return embed

        raise ValueError(f"Unknown embed_type: {self.embed_type}")




class EHR_module(nn.Module):
    def __init__(self, embed_size=256, hidden_size=256, device='cpu'):
        super(EHR_module, self).__init__()
        self.device = device
        self.demo_embed = EHR_Embedding(embed_type='demo', embed_size=embed_size, device=device)
        self.chart_embed = EHR_Embedding(embed_type='chart', varible_num=9, embed_size=embed_size, device=device)
        self.lab_embed = EHR_Embedding(embed_type='lab', varible_num=22, embed_size=embed_size, device=device)
        self.procedure_embed = EHR_Embedding(embed_type='procedure', varible_num=10, embed_size=embed_size,
                                             device=device)
        self.time_embed = EHR_Embedding(embed_type='time', embed_size=embed_size, device=device)

        self.lstm = nn.LSTM(input_size=2 * embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.cls_head = nn.Linear(2 * hidden_size, 2)

    def embedding(self, demo, chart, lab, procedure, time, pooling=False):
        demo_embeds = self.demo_embed(demo)
        embeds = []
        for idx, (chart_ts, lab_ts, procedure_ts) in enumerate(zip(chart, lab, procedure)):
            demo_embed = demo_embeds[idx]
            demo_embed = self.pooling(demo_embed.T.unsqueeze(0)).reshape(1, -1)
            embed_ts = []
            for ce, le, pe in zip(chart_ts, lab_ts, procedure_ts):
                ce_embed = self.chart_embed(ce)
                le_embed = self.lab_embed(le)
                pe_embed = self.procedure_embed(pe)
                embed_thistime = torch.cat([demo_embed, ce_embed, le_embed, pe_embed], dim=0)

                embed_thistime = torch.nan_to_num(
                    embed_thistime, nan=0.0, posinf=0.0, neginf=0.0
                )

                embed_thistime = self.pooling(embed_thistime.T.unsqueeze(0)).reshape(-1)
                embed_ts.append(embed_thistime)
            if embed_ts == []:

                embed_ts = self.pooling(demo_embed.T.unsqueeze(0)).reshape(1, -1)
            else:
                embed_ts = torch.stack(embed_ts)

            time_embed = self.time_embed(time[idx])
            embed_ts = torch.cat([embed_ts, time_embed], dim=1)


            embeds.append(embed_ts)

        embeds = pack_sequence(embeds, enforce_sorted=False)
        embeds, (_, _) = self.lstm(embeds)
        embeds, lengths = pad_packed_sequence(embeds, batch_first=True)

        embeds = list(map(lambda x: x[0][:x[1], :], zip(embeds, lengths)))
        if pooling:
            embeds = list(map(lambda x: self.pooling(x.T.unsqueeze(0)).reshape(-1), embeds))
            embeds = torch.stack(embeds)
        return embeds

    def forward(self, demo, chart, lab, procedure, time):
        embed = self.embedding(demo, chart, lab, procedure, time, pooling=True)
        pred = self.cls_head(embed.float())
        return pred

