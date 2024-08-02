from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torchcrf import CRF


def _align_labels_with_tokens(labels: list[str],
                              mask: list[bool],
                              label_all_tokens=False) -> list[str]:
    """ 为 tokens 对齐 labels.

    Args:
        labels (list[str]): 数据集 labels 字段.
        mask (list[bool]): 需要特殊对齐的位置设置为 True.
        label_all_tokens (bool, optional): 若为 True, 多个 token 对齐同一个 label. 否则 只有第一个 token 对齐, 其余 label 为 X. Defaults to False.

    Returns:
        list[str]: labels of the tokens.
    """
    new_labels = []
    word_index = 0

    for m in mask:
        if m:
            new_labels.append(labels[word_index])
            word_index += 1
        else:
            new_labels.append(
                "X") if not label_all_tokens else new_labels.append(
                    labels[word_index - 1])
    return new_labels


class Data(Dataset):

    def __init__(self, model_path: str, data: list[dict], seq_max_len: int,
                 label_vocab: list[str]):
        """ 序列标注数据集.

        Args:
            model_path (str): tokenizer 模型路径.
            data (list(DataInput)): 列表元素为 dict, 每个 dict 包含 sentence 和 labels(可选) 两个字段.
            seq_max_len (int): 句子的最大长度.
            label_vocab (list[str]): 所有 label 的列表.
        """
        self.data = data
        self.seq_max_len = seq_max_len
        self.label_vocab = label_vocab
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        sentence = self.data[index]['sentence']
        tokens = self.tokenizer.tokenize(sentence)
        encoding = self.tokenizer.encode_plus(
            sentence,
            truncation=True,
            add_special_tokens=False,  # 不添加 [CLS] 和 [SEP]
            padding='max_length',
            max_length=self.seq_max_len,
            return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze().tolist()
        mask = [False] * self.seq_max_len
        for idx, token in enumerate(tokens):
            if not token.startswith("##"):
                mask[idx] = True

        res = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'mask': torch.tensor(mask)
        }

        if (labels := self.data[index].get('labels')) is not None:
            res['origin_label_ids'] = torch.tensor(
                [self.label_vocab.index(label) for label in labels])

            # 使用 ignore index -100
            label_ids = [-100] * self.seq_max_len
            aligned_labels = _align_labels_with_tokens(labels=labels,
                                                       mask=mask)
            label_idx = 0
            for token_id_idx, token_id in enumerate(input_ids):
                # special token 对应的 label_id 为 -100.
                if token_id == self.tokenizer.pad_token_id:
                    continue
                # 将 labels 转换为 label_ids.
                # id 为 label 在 label_vocab中的下标.
                label = aligned_labels[label_idx]
                if label in self.label_vocab:  # 可能为 X 则相应的 label_id 还是为 -100
                    label_ids[token_id_idx] = self.label_vocab.index(label)
                label_idx += 1
            res['label_ids'] = torch.tensor(label_ids)
        return res


class Model(nn.Module):

    def __init__(self,
                 model_path: str,
                 embedding_dim: int,
                 hidden_dim: int,
                 label_vocab_len: int,
                 dropout=0.1) -> None:
        """ Bert + BiLSTM + CRF

        Args:
            model_path (str): bert 模型路径.
            embedding_dim (int): 模型嵌入维度.
            hidden_dim (int): 中间层维度.
            label_vocab_len (int): label 种类.
            dropout (float, optional): dropout 率. Defaults to 0.1.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.bilstm = nn.LSTM(embedding_dim,
                              hidden_dim // 2,
                              num_layers=2,
                              bidirectional=True,
                              batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, label_vocab_len)
        self.crf = CRF(label_vocab_len, batch_first=True)

    def forward(self, input_ids, attention_mask, mask, label_ids=None):
        # size 全部为tensor: (batch_size, seq_len). 其中 seq_len 全部为补全到 seq_max_len.
        embeds = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # size: (batch_size, seq_len, embedding_dim)
        enc, _ = self.bilstm(embeds)
        enc = self.dropout(enc)
        outputs = self.linear(enc)  # size: (batch_size, seq_len, num_tags)
        if label_ids is not None:
            label_ids[label_ids ==
                      -100] = 0  # 虽然设置了mask 这里 CRF 还是要将所有的 label 设置在范围值内
            # 这里和设置 mask = labels != -100 或者 mask = input_ids != 0 不同
            # 因为 label_all_tokens=False 后除了第一个子词外其余也要被忽略, 其 label_id 也为 -100
            loss = -self.crf.forward(
                outputs, label_ids, reduction='mean', mask=mask)
            return loss
        else:
            return torch.tensor(self.crf.decode(outputs, mask=mask))


if __name__ == '__main__':
    from pprint import pprint

    model_path = 'model/hfl/chinese-bert-wwm-ext'
    embedding_dim = 768
    label_vocab = ['O', 'B', 'I']
    data = [{
        'sentence': 'was at various times',
        'labels': ['O', 'O', 'B', 'I']
    }]  # was at var ##ious time
    test_data = [{
        'sentence': 'was at various times',
        'labels': ['O', 'O', 'B', 'I']
    }]
    batch_size = 1
    seq_max_len = 10  # 模型上下文长度最大为 512 也就只能补全到此

    model = Model(model_path=model_path,
                  embedding_dim=embedding_dim,
                  hidden_dim=100,
                  label_vocab_len=len(label_vocab))
    train_dataloader = DataLoader(dataset=Data(data=data,
                                               seq_max_len=seq_max_len,
                                               label_vocab=label_vocab,
                                               model_path=model_path),
                                  batch_size=batch_size)
    test_dataloader = DataLoader(dataset=Data(data=test_data,
                                              seq_max_len=seq_max_len,
                                              label_vocab=label_vocab,
                                              model_path=model_path),
                                 batch_size=batch_size)
    # for data_dict in train_dataloader:
    #     loss = model(data_dict['input_ids'], data_dict['attention_mask'], data_dict['mask'], data_dict['label_ids'])
    #     print(loss)

    for data_dict in test_dataloader:
        pprint(data_dict)
        y = data_dict['origin_label_ids']
        y_hat = model(data_dict['input_ids'], data_dict['attention_mask'],
                      data_dict['mask'])
        print(y)
        print(y_hat)
