import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Note_module(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512, device='cpu', dm=1, min_count=1, epochs=20, window=5, workers=4, negative=5):
        super(Note_module, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embed_size = embed_size


        self.doc2vec = Doc2Vec(
            vector_size=embed_size,
            dm=dm,
            min_count=min_count,
            epochs=epochs,
            window=window,
            workers=workers,
            negative=negative
        )
        self.doc2vec_ready = False

        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.cls_head = nn.Linear(hidden_size, 2)

    def _normalize_note(self, note):
        if isinstance(note, str):
            note = note.strip()
            return note.split() if len(note) > 0 else ["[empty]"]

        if isinstance(note, (list, tuple)):
            tokens = [str(x).strip() for x in note if str(x).strip() != ""]
            return tokens if len(tokens) > 0 else ["[empty]"]

        return [str(note)]

    def build_doc2vec_from_notes(self, all_notes):
        documents = []
        doc_id = 0

        for sample_notes in all_notes:
            if not isinstance(sample_notes, (list, tuple)):
                sample_notes = [sample_notes]

            for note in sample_notes:
                tokens = self._normalize_note(note)
                documents.append(TaggedDocument(words=tokens, tags=[str(doc_id)]))
                doc_id += 1

        if len(documents) == 0:
            raise ValueError("build_doc2vec_from_notes(): 没有收集到任何 notes")

        self.doc2vec.build_vocab(documents)
        self.doc2vec.train(
            documents,
            total_examples=self.doc2vec.corpus_count,
            epochs=self.doc2vec.epochs
        )
        self.doc2vec_ready = True
        print(f"[Note_module] Doc2Vec trained from scratch. docs={len(documents)}")

    def embedding(self, notes):

        if not self.doc2vec_ready:
            self.build_doc2vec_from_notes(notes)  # Train Doc2Vec if not ready

        vecs = []
        for sample_notes in notes:
            sample_vecs = []

            if not isinstance(sample_notes, (list, tuple)):
                sample_notes = [sample_notes]

            for note in sample_notes:
                tokens = self._normalize_note(note)
                vec = self.doc2vec.infer_vector(tokens)
                sample_vecs.append(vec.astype(np.float32))

            if len(sample_vecs) == 0:
                sample_vecs.append(np.zeros(self.embed_size, dtype=np.float32))

            vecs.append(np.array(sample_vecs, dtype=np.float32))

        embeds = [
            torch.tensor(vec, dtype=torch.float32, requires_grad=False, device=self.device)
            for vec in vecs
        ]

        embeds = pack_sequence(embeds, enforce_sorted=False)
        embeds, (_, _) = self.lstm(embeds)
        embeds, lengths = pad_packed_sequence(embeds, batch_first=True)
        embeds = [e[:l, :] for e, l in zip(embeds, lengths)]

        embeds = [self.pooling(e.T.unsqueeze(0)).reshape(-1) for e in embeds]
        embeds = torch.stack(embeds)
        return embeds

    def forward(self, notes):
        embeds = self.embedding(notes)
        preds = self.cls_head(embeds.float())
        return preds