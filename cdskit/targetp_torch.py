import copy
import json
import os
import time

import numpy as np

from cdskit.localize_bilstm import require_torch, resolve_torch_device
from cdskit.localize_model import (
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    normalize_organism_group,
    softmax,
)


TARGETP_TORCH_MODEL_TYPE = 'targetp_torch_v1'
TARGETP_BLOSUM62_ORDER = 'ARNDCQEGHILKMFPSTWYV'
TARGETP_SIGNAL_CLASS_TO_HEAD = {
    1: 0,  # SP
    2: 1,  # mTP
    3: 2,  # cTP
    4: 3,  # lTP / thylakoid luminal transit peptide
}
TARGETP_TORCH_DEFAULTS = {
    'seq_len': 200,
    'n_input': 20,
    'hidden_rnn': 256,
    'n_filters': 32,
    'hidden_fc': 256,
    'filter_size': 1,
    'n_attention': 13,
    'attention_size': 144,
    'input_keep_prob': 0.75,
    'encoder_keep_prob': 0.6,
    'rnn_keep_prob': 0.5,
    'learning_rate': 0.002,
    'lr_gamma': 0.1,
    'batch_size': 64,
    'epochs': 150,
    'max_lr_reductions': 5,
    'patience_epochs': 8,
    'type_class_weight': 'none',
    'selection_metric': 'val_loss',
    'balanced_batch': 'no',
    'initializer': 'targetp_tf',
    'grad_clip_norm': 0.0,
    'rnn_impl': 'targetp_tf_cell',
}


BLOSUM62 = np.asarray([
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
], dtype=np.float32)

TARGETP_BLOSUM62_PROBABILITIES = np.asarray([
    [0.2901484, 0.0310391, 0.0256410, 0.0296896, 0.0215924, 0.0256410, 0.0404858, 0.0782726, 0.0148448, 0.0431849, 0.0593792, 0.0445344, 0.0175439, 0.0215924, 0.0296896, 0.0850202, 0.0499325, 0.0053981, 0.0175439, 0.0688259],
    [0.0445736, 0.3449612, 0.0387597, 0.0310078, 0.0077519, 0.0484496, 0.0523256, 0.0329457, 0.0232558, 0.0232558, 0.0465116, 0.1201550, 0.0155039, 0.0174419, 0.0193798, 0.0445736, 0.0348837, 0.0058140, 0.0174419, 0.0310078],
    [0.0426966, 0.0449438, 0.3168540, 0.0831461, 0.0089888, 0.0337079, 0.0494382, 0.0651685, 0.0314607, 0.0224719, 0.0314607, 0.0539326, 0.0112360, 0.0179775, 0.0202247, 0.0696629, 0.0494382, 0.0044944, 0.0157303, 0.0269663],
    [0.0410448, 0.0298507, 0.0690298, 0.3973881, 0.0074627, 0.0298507, 0.0914179, 0.0466418, 0.0186567, 0.0223881, 0.0279851, 0.0447761, 0.0093284, 0.0149254, 0.0223881, 0.0522388, 0.0354478, 0.0037313, 0.0111940, 0.0242537],
    [0.0650406, 0.0162602, 0.0162602, 0.0162602, 0.4837398, 0.0121951, 0.0162602, 0.0325203, 0.0081301, 0.0447154, 0.0650406, 0.0203252, 0.0162602, 0.0203252, 0.0162602, 0.0406504, 0.0365854, 0.0040650, 0.0121951, 0.0569106],
    [0.0558824, 0.0735294, 0.0441176, 0.0470588, 0.0088235, 0.2147059, 0.1029412, 0.0411765, 0.0294118, 0.0264706, 0.0470588, 0.0911765, 0.0205882, 0.0147059, 0.0235294, 0.0558824, 0.0411765, 0.0058824, 0.0205882, 0.0352941],
    [0.0552486, 0.0497238, 0.0405157, 0.0902394, 0.0073665, 0.0644567, 0.2965009, 0.0349908, 0.0257827, 0.0220994, 0.0368324, 0.0755064, 0.0128913, 0.0165746, 0.0257827, 0.0552486, 0.0368324, 0.0055249, 0.0165746, 0.0313076],
    [0.0782726, 0.0229420, 0.0391363, 0.0337382, 0.0107962, 0.0188934, 0.0256410, 0.5101214, 0.0134953, 0.0188934, 0.0283401, 0.0337382, 0.0094467, 0.0161943, 0.0188934, 0.0512821, 0.0296896, 0.0053981, 0.0107962, 0.0242915],
    [0.0419847, 0.0458015, 0.0534351, 0.0381679, 0.0076336, 0.0381679, 0.0534351, 0.0381679, 0.3549618, 0.0229008, 0.0381679, 0.0458015, 0.0152672, 0.0305344, 0.0190840, 0.0419847, 0.0267176, 0.0076336, 0.0572519, 0.0229008],
    [0.0471281, 0.0176730, 0.0147275, 0.0176730, 0.0162003, 0.0132548, 0.0176730, 0.0206186, 0.0088365, 0.2709867, 0.1678940, 0.0235641, 0.0368188, 0.0441826, 0.0147275, 0.0250368, 0.0397644, 0.0058910, 0.0206186, 0.1767305],
    [0.0445344, 0.0242915, 0.0141700, 0.0151822, 0.0161943, 0.0161943, 0.0202429, 0.0212551, 0.0101215, 0.1153846, 0.3755061, 0.0253036, 0.0495951, 0.0546559, 0.0141700, 0.0242915, 0.0334008, 0.0070850, 0.0222672, 0.0961538],
    [0.0569948, 0.1070812, 0.0414508, 0.0414508, 0.0086356, 0.0535406, 0.0708117, 0.0431779, 0.0207254, 0.0276338, 0.0431779, 0.2780656, 0.0155440, 0.0155440, 0.0276338, 0.0535406, 0.0397237, 0.0051813, 0.0172712, 0.0328152],
    [0.0522088, 0.0321285, 0.0200803, 0.0200803, 0.0160643, 0.0281124, 0.0281124, 0.0281124, 0.0160643, 0.1004016, 0.1967872, 0.0361446, 0.1606426, 0.0481928, 0.0160643, 0.0361446, 0.0401606, 0.0080321, 0.0240964, 0.0923695],
    [0.0338266, 0.0190275, 0.0169133, 0.0169133, 0.0105708, 0.0105708, 0.0190275, 0.0253700, 0.0169133, 0.0634249, 0.1141649, 0.0190275, 0.0253700, 0.3868922, 0.0105708, 0.0253700, 0.0253700, 0.0169133, 0.0887949, 0.0549683],
    [0.0568475, 0.0258398, 0.0232558, 0.0310078, 0.0103359, 0.0206718, 0.0361757, 0.0361757, 0.0129199, 0.0258398, 0.0361757, 0.0413437, 0.0103359, 0.0129199, 0.4935400, 0.0439276, 0.0361757, 0.0025840, 0.0129199, 0.0310078],
    [0.1099476, 0.0401396, 0.0541012, 0.0488656, 0.0174520, 0.0331588, 0.0523560, 0.0663176, 0.0191972, 0.0296684, 0.0418848, 0.0541012, 0.0157068, 0.0209424, 0.0296684, 0.2198953, 0.0820244, 0.0052356, 0.0174520, 0.0418848],
    [0.0729783, 0.0355030, 0.0433925, 0.0374753, 0.0177515, 0.0276134, 0.0394477, 0.0433925, 0.0138067, 0.0532544, 0.0650888, 0.0453649, 0.0197239, 0.0236686, 0.0276134, 0.0927022, 0.2465483, 0.0059172, 0.0177515, 0.0710059],
    [0.0303030, 0.0227273, 0.0151515, 0.0151515, 0.0075758, 0.0151515, 0.0227273, 0.0303030, 0.0151515, 0.0303030, 0.0530303, 0.0227273, 0.0151515, 0.0606061, 0.0075758, 0.0227273, 0.0227273, 0.4924242, 0.0681818, 0.0303030],
    [0.0404984, 0.0280374, 0.0218069, 0.0186916, 0.0093458, 0.0218069, 0.0280374, 0.0249221, 0.0467290, 0.0436137, 0.0685358, 0.0311526, 0.0186916, 0.1308411, 0.0155763, 0.0311526, 0.0280374, 0.0280374, 0.3177570, 0.0467290],
    [0.0699588, 0.0219479, 0.0164609, 0.0178326, 0.0192044, 0.0164609, 0.0233196, 0.0246914, 0.0082305, 0.1646091, 0.1303155, 0.0260631, 0.0315501, 0.0356653, 0.0164609, 0.0329218, 0.0493827, 0.0054870, 0.0205761, 0.2688614],
], dtype=np.float32)


def targetp_blosum62_probability_table():
    weights = TARGETP_BLOSUM62_PROBABILITIES.astype(np.float64)
    table = {
        TARGETP_BLOSUM62_ORDER[i]: weights[i, :].astype(np.float32)
        for i in range(len(TARGETP_BLOSUM62_ORDER))
    }
    uniform = np.full((len(TARGETP_BLOSUM62_ORDER),), 1.0 / len(TARGETP_BLOSUM62_ORDER), dtype=np.float32)
    table['B'] = uniform
    table['J'] = uniform
    table['U'] = table['C']
    table['X'] = uniform
    table['Z'] = uniform
    table['<PAD>'] = uniform
    return table


def organism_group_to_targetp_org(organism_group):
    group = normalize_organism_group(organism_group)
    return 1 if group == 'plant' else 0


def _limit_torch_threads(torch):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _to_targetp_blosum_sequence(aa_seq):
    seq = str(aa_seq).upper().replace(' ', '')
    if seq.endswith('*'):
        seq = seq[:-1]
    if '*' in seq:
        raise Exception('Internal stop codon detected in translated peptide sequence. Exiting.')
    allowed = set(TARGETP_BLOSUM62_ORDER + 'BJUXZ')
    out = list()
    for ch in seq:
        if ch in allowed:
            out.append(ch)
        else:
            out.append('X')
    return ''.join(out)


def encode_targetp_blosum_sequences(aa_sequences, seq_len=200, blosum_table=None):
    seq_len = int(seq_len)
    if seq_len < 1:
        raise ValueError('seq_len should be >= 1.')
    table = targetp_blosum62_probability_table() if blosum_table is None else blosum_table
    pad_vec = np.asarray(table['<PAD>'], dtype=np.float32)
    x = np.zeros((len(aa_sequences), seq_len, len(TARGETP_BLOSUM62_ORDER)), dtype=np.float32)
    lengths = np.zeros((len(aa_sequences),), dtype=np.int64)
    for row_i, aa_seq in enumerate(aa_sequences):
        seq = _to_targetp_blosum_sequence(aa_seq)
        use_len = min(len(seq), seq_len)
        lengths[row_i] = max(1, use_len)
        if use_len > 0:
            for pos_i in range(use_len):
                x[row_i, pos_i, :] = np.asarray(table.get(seq[pos_i], table['X']), dtype=np.float32)
        if use_len < seq_len:
            x[row_i, use_len:, :] = pad_vec
    return x, lengths


def _macro_f1_from_indices(true_idx, pred_idx, num_class):
    true_idx = np.asarray(true_idx, dtype=np.int64)
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    f1_values = list()
    for class_i in range(int(num_class)):
        tp = float(np.sum((true_idx == class_i) & (pred_idx == class_i)))
        fp = float(np.sum((true_idx != class_i) & (pred_idx == class_i)))
        fn = float(np.sum((true_idx == class_i) & (pred_idx != class_i)))
        precision = 0.0 if (tp + fp) <= 0.0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) <= 0.0 else tp / (tp + fn)
        if (precision + recall) <= 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append((2.0 * precision * recall) / (precision + recall))
    return float(np.mean(np.asarray(f1_values, dtype=np.float64)))


def _build_targetp2_torch_module(
    torch,
    nn,
    seq_len=200,
    n_input=20,
    hidden_rnn=256,
    n_filters=32,
    hidden_fc=256,
    filter_size=1,
    n_attention=13,
    attention_size=144,
    n_type=5,
    n_org=2,
    input_keep_prob=0.75,
    encoder_keep_prob=0.6,
    rnn_impl='torch_lstm',
):
    if int(n_attention) < 4:
        raise ValueError('n_attention should be >= 4 for TargetP cleavage-site heads.')
    rnn_impl = str(rnn_impl or 'torch_lstm').strip().lower()
    if rnn_impl not in ['torch_lstm', 'targetp_tf_cell']:
        raise ValueError('Unsupported TargetP torch rnn_impl: {}'.format(rnn_impl))

    class TargetP2TorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_len = int(seq_len)
            self.input_keep_prob = float(input_keep_prob)
            self.encoder_keep_prob = float(encoder_keep_prob)
            self.rnn_impl = rnn_impl
            self.conv = nn.Conv1d(
                in_channels=int(n_input),
                out_channels=int(n_filters),
                kernel_size=int(filter_size),
                padding=int(filter_size) // 2,
            )
            self.conv_dropout = nn.Dropout(p=max(0.0, min(1.0, 1.0 - float(encoder_keep_prob))))
            self.organism_fc = nn.Linear(int(n_org), int(hidden_rnn))
            if self.rnn_impl == 'targetp_tf_cell':
                self.fw_cell = nn.LSTMCell(input_size=int(n_filters), hidden_size=int(hidden_rnn))
                self.bw_cell = nn.LSTMCell(input_size=int(n_filters), hidden_size=int(hidden_rnn))
            else:
                self.encoder = nn.LSTM(
                    input_size=int(n_filters),
                    hidden_size=int(hidden_rnn),
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            self.attention_pre = nn.Conv1d(
                in_channels=int(hidden_rnn) * 2,
                out_channels=int(attention_size),
                kernel_size=1,
            )
            self.attention_align = nn.Conv1d(
                in_channels=int(attention_size),
                out_channels=int(n_attention),
                kernel_size=1,
            )
            self.fc = nn.Linear(int(n_attention) * int(hidden_rnn) * 2, int(hidden_fc))
            self.type_classifier = nn.Linear(int(hidden_fc), int(n_type))

        def _input_dropout(self, x):
            if (not self.training) or self.input_keep_prob >= 1.0:
                return x
            keep = max(1.0e-6, min(1.0, self.input_keep_prob))
            mask = torch.empty((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype)
            mask.bernoulli_(keep)
            mask = mask / keep
            return x * mask

        def _dropout_keep(self, value, keep):
            keep = max(1.0e-6, min(1.0, float(keep)))
            if (not self.training) or keep >= 1.0:
                return value
            mask = torch.empty_like(value)
            mask.bernoulli_(keep)
            return value * (mask / keep)

        def _run_tf_style_lstm_cell(self, conv, lengths, org_state, rnn_keep_prob):
            batch_size = int(conv.shape[0])
            hidden_size = int(org_state.shape[1])
            zero_out = torch.zeros((batch_size, hidden_size), dtype=conv.dtype, device=conv.device)

            def _run_direction(cell, time_indices):
                h = org_state
                c = org_state
                outputs = [zero_out for _ in range(self.seq_len)]
                for pos_i in time_indices:
                    active = (lengths > int(pos_i)).to(dtype=conv.dtype).unsqueeze(1)
                    next_h, next_c = cell(conv[:, int(pos_i), :], (h, c))
                    dropped_c = self._dropout_keep(next_c, rnn_keep_prob)
                    dropped_h = self._dropout_keep(next_h, rnn_keep_prob)
                    dropped_output = self._dropout_keep(next_h, rnn_keep_prob)
                    h = (active * dropped_h) + ((1.0 - active) * h)
                    c = (active * dropped_c) + ((1.0 - active) * c)
                    outputs[int(pos_i)] = active * dropped_output
                return torch.stack(outputs, dim=1)

            fw = _run_direction(self.fw_cell, range(self.seq_len))
            bw = _run_direction(self.bw_cell, range(self.seq_len - 1, -1, -1))
            return torch.cat([fw, bw], dim=2)

        def _run_encoder(self, conv, lengths, org_state, rnn_keep_prob):
            if self.rnn_impl == 'targetp_tf_cell':
                return self._run_tf_style_lstm_cell(
                    conv=conv,
                    lengths=lengths,
                    org_state=org_state,
                    rnn_keep_prob=rnn_keep_prob,
                )
            h0 = torch.stack([org_state, org_state], dim=0).contiguous()
            c0 = torch.stack([org_state, org_state], dim=0).contiguous()
            packed = nn.utils.rnn.pack_padded_sequence(
                conv,
                lengths=lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.encoder(packed, (h0, c0))
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=self.seq_len,
            )
            if self.training and float(rnn_keep_prob) < 1.0:
                import torch.nn.functional as F

                encoded = F.dropout(
                    encoded,
                    p=max(0.0, min(1.0, 1.0 - float(rnn_keep_prob))),
                    training=True,
                )
            return encoded

        def forward(self, x, lengths, organism, rnn_keep_prob=1.0):
            import torch.nn.functional as F

            lengths = lengths.long().clamp(min=1, max=self.seq_len)
            x = self._input_dropout(x)
            conv_in = x.transpose(1, 2)
            conv = F.elu(self.conv(conv_in)).transpose(1, 2)
            conv = self.conv_dropout(conv)

            org_hot = F.one_hot(organism.long().clamp(min=0, max=1), num_classes=2).to(dtype=x.dtype)
            org_state = F.relu(self.organism_fc(org_hot))
            encoded = self._run_encoder(
                conv=conv,
                lengths=lengths,
                org_state=org_state,
                rnn_keep_prob=rnn_keep_prob,
            )

            att_in = encoded.transpose(1, 2)
            att_pre = torch.tanh(self.attention_pre(att_in))
            align = self.attention_align(att_pre).transpose(1, 2)
            mask = torch.arange(self.seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            masked_align = align.masked_fill(~mask.unsqueeze(2), -100000.0)
            alpha = torch.softmax(masked_align, dim=1)
            weighted = encoded.unsqueeze(2) * alpha.unsqueeze(3)
            weighted_sum = weighted.sum(dim=1)
            flat = weighted_sum.reshape(weighted_sum.shape[0], -1)
            hidden = F.elu(self.fc(flat))
            type_logits = self.type_classifier(hidden)
            return {
                'type_logits': type_logits,
                'attention_logits': masked_align,
                'hidden': hidden,
            }

    return TargetP2TorchNet()


def initialize_targetp2_torch_module(torch, nn, module, initializer='targetp_tf'):
    initializer = str(initializer or 'targetp_tf').strip().lower()
    if initializer == 'pytorch':
        return module
    if initializer != 'targetp_tf':
        raise ValueError('Unsupported TargetP torch initializer: {}'.format(initializer))
    def _initialize_lstm_param(name, param):
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            hidden = int(param.shape[1])
            for start in range(0, int(param.shape[0]), hidden):
                nn.init.orthogonal_(param[start:start + hidden])
        elif 'bias' in name:
            nn.init.zeros_(param)
            if 'bias_ih' in name:
                hidden = int(param.shape[0] // 4)
                with torch.no_grad():
                    param[hidden:2 * hidden].fill_(1.0)

    for child in module.modules():
        if isinstance(child, nn.Conv1d) or isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LSTM) or isinstance(child, nn.LSTMCell):
            for name, param in child.named_parameters():
                _initialize_lstm_param(name=name, param=param)
    return module


def _targetp2_loss(torch, outputs, y_type, y_cs, type_weight=None):
    import torch.nn.functional as F

    type_logits = outputs['type_logits']
    align = outputs['attention_logits']
    loss = F.cross_entropy(type_logits, y_type.long(), weight=type_weight)
    cs_pos = torch.argmax(y_cs.long(), dim=1)
    batch_size = max(1, int(y_type.shape[0]))
    for class_idx, head_idx in TARGETP_SIGNAL_CLASS_TO_HEAD.items():
        class_mask = y_type.long() == int(class_idx)
        if torch.any(class_mask):
            per_sample = F.cross_entropy(
                align[:, :, int(head_idx)],
                cs_pos,
                reduction='none',
            )
            loss = loss + ((per_sample * class_mask.to(dtype=per_sample.dtype)).sum() / float(batch_size))
    return loss


def _iter_minibatches(num_rows, batch_size, rng, shuffle=True):
    indices = np.arange(int(num_rows), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, indices.shape[0], int(batch_size)):
        yield indices[start:start + int(batch_size)]


def _init_balanced_pools(y_train, rng):
    y_train = np.asarray(y_train, dtype=np.int64)
    classes = sorted(np.unique(y_train).astype(int).tolist())
    pools = dict()
    cursors = dict()
    for class_idx in classes:
        pool = np.where(y_train == int(class_idx))[0].astype(np.int64)
        rng.shuffle(pool)
        pools[int(class_idx)] = pool
        cursors[int(class_idx)] = 0
    return classes, pools, cursors


def _draw_balanced_index(class_idx, pools, cursors, rng):
    pool = pools[int(class_idx)]
    cursor = int(cursors[int(class_idx)])
    if cursor >= pool.shape[0]:
        pool = pool.copy()
        rng.shuffle(pool)
        pools[int(class_idx)] = pool
        cursor = 0
    out = int(pool[cursor])
    cursors[int(class_idx)] = cursor + 1
    return out


def _sample_balanced_batch(y_train, batch_size, rng, pools, cursors, classes):
    batch = list()
    class_order = np.asarray(classes, dtype=np.int64)
    rng.shuffle(class_order)
    for class_idx in class_order.tolist():
        if len(batch) >= int(batch_size):
            break
        batch.append(_draw_balanced_index(class_idx, pools, cursors, rng))
    while len(batch) < int(batch_size):
        class_idx = int(classes[int(rng.integers(0, len(classes)))])
        batch.append(_draw_balanced_index(class_idx, pools, cursors, rng))
    batch = np.asarray(batch, dtype=np.int64)
    rng.shuffle(batch)
    return batch


def _predict_encoded_with_module(torch, module, device, x, lengths, org, batch_size):
    module.eval()
    probs = list()
    with torch.no_grad():
        for batch_idx in _iter_minibatches(x.shape[0], batch_size, np.random.default_rng(1), shuffle=False):
            xb = torch.as_tensor(x[batch_idx], dtype=torch.float32, device=device)
            lb = torch.as_tensor(lengths[batch_idx], dtype=torch.long, device=device)
            ob = torch.as_tensor(org[batch_idx], dtype=torch.long, device=device)
            outputs = module(x=xb, lengths=lb, organism=ob, rnn_keep_prob=1.0)
            prob = torch.softmax(outputs['type_logits'], dim=1).detach().cpu().numpy()
            probs.append(prob)
    if len(probs) == 0:
        return np.zeros((0, len(LOCALIZATION_CLASSES)), dtype=np.float32)
    return np.vstack(probs).astype(np.float32)


def _evaluate_encoded(torch, module, device, x, y_type, y_cs, lengths, org, batch_size, type_weight=None):
    module.eval()
    losses = list()
    probs = list()
    with torch.no_grad():
        for batch_idx in _iter_minibatches(x.shape[0], batch_size, np.random.default_rng(1), shuffle=False):
            xb = torch.as_tensor(x[batch_idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_type[batch_idx], dtype=torch.long, device=device)
            cb = torch.as_tensor(y_cs[batch_idx], dtype=torch.long, device=device)
            lb = torch.as_tensor(lengths[batch_idx], dtype=torch.long, device=device)
            ob = torch.as_tensor(org[batch_idx], dtype=torch.long, device=device)
            outputs = module(x=xb, lengths=lb, organism=ob, rnn_keep_prob=1.0)
            loss = _targetp2_loss(
                torch=torch,
                outputs=outputs,
                y_type=yb,
                y_cs=cb,
                type_weight=type_weight,
            )
            losses.append(float(loss.detach().cpu().item()))
            probs.append(torch.softmax(outputs['type_logits'], dim=1).detach().cpu().numpy())
    prob = np.vstack(probs).astype(np.float32) if len(probs) > 0 else np.zeros((0, len(LOCALIZATION_CLASSES)))
    pred = np.argmax(prob, axis=1).astype(np.int64) if prob.shape[0] > 0 else np.zeros((0,), dtype=np.int64)
    return {
        'loss': float(np.mean(np.asarray(losses, dtype=np.float64))) if len(losses) > 0 else 0.0,
        'macro_f1': _macro_f1_from_indices(y_type, pred, len(LOCALIZATION_CLASSES)) if prob.shape[0] > 0 else 0.0,
        'accuracy': float(np.mean(pred == np.asarray(y_type, dtype=np.int64))) if prob.shape[0] > 0 else 0.0,
        'prob_matrix': prob,
    }


def _state_dict_to_cpu(module):
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def _rnn_impl_from_state_dict(state):
    if not isinstance(state, dict):
        return ''
    keys = [str(key) for key in state.keys()]
    if any(key.startswith('encoder.') for key in keys):
        return 'torch_lstm'
    if any(key.startswith('fw_cell.') or key.startswith('bw_cell.') for key in keys):
        return 'targetp_tf_cell'
    return ''


def _payload_rnn_impl_from_state(payload):
    if not isinstance(payload, dict):
        return ''
    state = payload.get('latest_state_dict', payload.get('state_dict'))
    return _rnn_impl_from_state_dict(state=state)


def _targetp_torch_payload(
    module,
    config,
    seed,
    best_state,
    history,
    best_metrics,
    final_val_metrics,
    device,
    training_complete,
    optimizer=None,
    best_epoch=0,
    current_lr=None,
    lr_reductions=0,
    rng=None,
    torch=None,
    latest_state=None,
):
    payload = {
        'model_type': TARGETP_TORCH_MODEL_TYPE,
        'class_order': list(LOCALIZATION_CLASSES),
        'config': dict(config),
        'seed': int(seed),
        'state_dict': best_state,
        'latest_state_dict': latest_state if latest_state is not None else _state_dict_to_cpu(module),
        'latest_epoch': int(len(history)),
        'history': list(history),
        'best_metrics': best_metrics,
        'final_val_metrics': final_val_metrics,
        'device': str(device),
        'training_complete': bool(training_complete),
        'best_epoch': int(best_epoch),
        'current_lr': float(current_lr if current_lr is not None else config.get('learning_rate', 0.0)),
        'lr_reductions': int(lr_reductions),
    }
    if optimizer is not None:
        payload['optimizer_state'] = optimizer.state_dict()
    if rng is not None:
        payload['rng_state'] = copy.deepcopy(rng.bit_generator.state)
    if torch is not None:
        payload['torch_rng_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload['torch_cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return payload


def _load_optimizer_state(optimizer, payload, device):
    state = payload.get('optimizer_state')
    if state is None:
        return
    optimizer.load_state_dict(state)
    for value in optimizer.state.values():
        for state_key, state_value in value.items():
            if hasattr(state_value, 'to'):
                value[state_key] = state_value.to(device)


def _can_resume_optimizer_state(payload):
    if not isinstance(payload, dict):
        return False
    if payload.get('optimizer_state') is None:
        return False
    history_len = len(payload.get('history', []))
    if 'latest_epoch' in payload:
        return int(payload.get('latest_epoch', -1)) == int(history_len)
    return not bool(payload.get('training_complete', False))


def fit_targetp2_torch_model(
    x_train,
    y_type_train,
    y_cs_train,
    len_train,
    org_train,
    x_val,
    y_type_val,
    y_cs_val,
    len_val,
    org_val,
    seed=1,
    device='auto',
    verbose=False,
    resume_payload=None,
    epoch_checkpoint_path='',
    **kwargs
):
    torch, nn = require_torch()
    _limit_torch_threads(torch)
    config = dict(TARGETP_TORCH_DEFAULTS)
    config.update({key: value for key, value in kwargs.items() if value is not None})
    if isinstance(resume_payload, dict):
        resume_config = resume_payload.get('config', {})
        resume_state_impl = _payload_rnn_impl_from_state(resume_payload)
        if resume_state_impl != '':
            config['rnn_impl'] = resume_state_impl
        elif 'rnn_impl' in resume_config and 'rnn_impl' not in kwargs:
            config['rnn_impl'] = resume_config['rnn_impl']
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    resolved_device = resolve_torch_device(device_text=device)
    module = _build_targetp2_torch_module(
        torch=torch,
        nn=nn,
        seq_len=int(config['seq_len']),
        n_input=int(config['n_input']),
        hidden_rnn=int(config['hidden_rnn']),
        n_filters=int(config['n_filters']),
        hidden_fc=int(config['hidden_fc']),
        filter_size=int(config['filter_size']),
        n_attention=int(config['n_attention']),
        attention_size=int(config['attention_size']),
        input_keep_prob=float(config['input_keep_prob']),
        encoder_keep_prob=float(config['encoder_keep_prob']),
        rnn_impl=str(config.get('rnn_impl', 'torch_lstm')),
    )
    initialize_targetp2_torch_module(
        torch=torch,
        nn=nn,
        module=module,
        initializer=config.get('initializer', 'targetp_tf'),
    )
    module.to(resolved_device)
    optimizer = torch.optim.Adam(module.parameters(), lr=float(config['learning_rate']))
    rng = np.random.default_rng(int(seed))
    type_weight = None
    weight_mode = str(config.get('type_class_weight', 'none')).strip().lower()
    if weight_mode not in ['none', 'balanced']:
        raise ValueError('type_class_weight should be none or balanced.')
    if weight_mode == 'balanced':
        counts = np.bincount(
            np.asarray(y_type_train, dtype=np.int64),
            minlength=len(LOCALIZATION_CLASSES),
        ).astype(np.float32)
        total = float(np.sum(counts))
        weights = np.zeros((len(LOCALIZATION_CLASSES),), dtype=np.float32)
        for class_i in range(len(LOCALIZATION_CLASSES)):
            if counts[class_i] > 0.0:
                weights[class_i] = total / (float(len(LOCALIZATION_CLASSES)) * float(counts[class_i]))
        type_weight = torch.as_tensor(weights, dtype=torch.float32, device=resolved_device)

    batch_size = int(config['batch_size'])
    epochs = int(config['epochs'])
    balanced_batch = str(config.get('balanced_batch', 'no')).strip().lower() in ['yes', 'true', '1']
    selection_metric = str(config.get('selection_metric', 'val_loss')).strip().lower()
    if selection_metric not in ['val_loss', 'val_macro_f1']:
        raise ValueError('selection_metric should be val_loss or val_macro_f1.')
    best_state = copy.deepcopy(_state_dict_to_cpu(module))
    best_metrics = None
    best_epoch = 0
    lr_reductions = 0
    history = list()
    current_lr = float(config['learning_rate'])
    start_epoch = 0
    if isinstance(resume_payload, dict):
        state = resume_payload.get('latest_state_dict', resume_payload.get('state_dict'))
        if state is not None:
            module.load_state_dict(state, strict=True)
        if resume_payload.get('state_dict') is not None:
            best_state = resume_payload['state_dict']
        best_metrics = resume_payload.get('best_metrics')
        best_epoch = int(resume_payload.get('best_epoch', 0) or 0)
        history = list(resume_payload.get('history', []))
        start_epoch = len(history)
        lr_reductions = int(resume_payload.get('lr_reductions', 0) or 0)
        current_lr = float(resume_payload.get('current_lr', current_lr))
        for group in optimizer.param_groups:
            group['lr'] = current_lr
        if _can_resume_optimizer_state(resume_payload):
            _load_optimizer_state(
                optimizer=optimizer,
                payload=resume_payload,
                device=resolved_device,
            )
        if resume_payload.get('rng_state') is not None:
            rng.bit_generator.state = resume_payload['rng_state']
        if resume_payload.get('torch_rng_state') is not None:
            torch.set_rng_state(resume_payload['torch_rng_state'])
        if torch.cuda.is_available() and resume_payload.get('torch_cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(resume_payload['torch_cuda_rng_state_all'])
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        module.train()
        train_losses = list()
        if balanced_batch:
            balanced_classes, pools, cursors = _init_balanced_pools(
                y_train=y_type_train,
                rng=rng,
            )
            n_batch = int(np.ceil(float(x_train.shape[0]) / float(batch_size)))
            batch_iter = (
                _sample_balanced_batch(
                    y_train=y_type_train,
                    batch_size=batch_size,
                    rng=rng,
                    pools=pools,
                    cursors=cursors,
                    classes=balanced_classes,
                )
                for _ in range(n_batch)
            )
        else:
            batch_iter = _iter_minibatches(x_train.shape[0], batch_size, rng, shuffle=True)
        for batch_idx in batch_iter:
            xb = torch.as_tensor(x_train[batch_idx], dtype=torch.float32, device=resolved_device)
            yb = torch.as_tensor(y_type_train[batch_idx], dtype=torch.long, device=resolved_device)
            cb = torch.as_tensor(y_cs_train[batch_idx], dtype=torch.long, device=resolved_device)
            lb = torch.as_tensor(len_train[batch_idx], dtype=torch.long, device=resolved_device)
            ob = torch.as_tensor(org_train[batch_idx], dtype=torch.long, device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = module(
                x=xb,
                lengths=lb,
                organism=ob,
                rnn_keep_prob=float(config['rnn_keep_prob']),
            )
            loss = _targetp2_loss(
                torch=torch,
                outputs=outputs,
                y_type=yb,
                y_cs=cb,
                type_weight=type_weight,
            )
            loss.backward()
            grad_clip_norm = float(config.get('grad_clip_norm', 0.0))
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_metrics = _evaluate_encoded(
            torch=torch,
            module=module,
            device=resolved_device,
            x=x_val,
            y_type=y_type_val,
            y_cs=y_cs_val,
            lengths=len_val,
            org=org_val,
            batch_size=batch_size,
            type_weight=type_weight,
        )
        train_loss = float(np.mean(np.asarray(train_losses, dtype=np.float64))) if len(train_losses) > 0 else 0.0
        row = {
            'epoch': int(epoch + 1),
            'train_loss': train_loss,
            'val_loss': float(val_metrics['loss']),
            'val_macro_f1': float(val_metrics['macro_f1']),
            'val_accuracy': float(val_metrics['accuracy']),
            'learning_rate': float(current_lr),
            'elapsed_sec': float(time.time() - start_time),
        }
        history.append(row)
        if bool(verbose):
            print(
                'epoch={epoch} train_loss={train_loss:.4f} '
                'val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f} '
                'val_acc={val_accuracy:.4f} lr={learning_rate:.6g}'.format(**row),
                flush=True,
            )
        is_better = False
        if best_metrics is None:
            is_better = True
        elif selection_metric == 'val_loss':
            is_better = row['val_loss'] < float(best_metrics['val_loss'])
        else:
            is_better = row['val_macro_f1'] > float(best_metrics['val_macro_f1'])
        if is_better:
            best_metrics = dict(row)
            best_epoch = int(epoch + 1)
            best_state = copy.deepcopy(_state_dict_to_cpu(module))
        if (epoch + 1) > 10 and (int(epoch + 1) - int(best_epoch)) == int(config['patience_epochs']):
            current_lr = current_lr * float(config['lr_gamma'])
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            lr_reductions += 1
            best_epoch = int(epoch + 1)
        if lr_reductions >= int(config['max_lr_reductions']):
            break
        if str(epoch_checkpoint_path or '').strip() != '':
            save_torch_payload(
                path=epoch_checkpoint_path,
                payload=_targetp_torch_payload(
                    module=module,
                    config=config,
                    seed=seed,
                    best_state=best_state,
                    history=history,
                    best_metrics=best_metrics,
                    final_val_metrics=None,
                    device=resolved_device,
                    training_complete=False,
                    optimizer=optimizer,
                    best_epoch=best_epoch,
                    current_lr=current_lr,
                    lr_reductions=lr_reductions,
                    rng=rng,
                    torch=torch,
                ),
            )

    latest_state = copy.deepcopy(_state_dict_to_cpu(module))
    module.load_state_dict(best_state, strict=True)
    final_val = _evaluate_encoded(
        torch=torch,
        module=module,
        device=resolved_device,
        x=x_val,
        y_type=y_type_val,
        y_cs=y_cs_val,
        lengths=len_val,
        org=org_val,
        batch_size=batch_size,
        type_weight=type_weight,
    )
    final_val_metrics = {
        'loss': float(final_val['loss']),
        'macro_f1': float(final_val['macro_f1']),
        'accuracy': float(final_val['accuracy']),
    }
    payload = _targetp_torch_payload(
        module=module,
        config=config,
        seed=seed,
        best_state=best_state,
        history=history,
        best_metrics=best_metrics,
        final_val_metrics=final_val_metrics,
        device=resolved_device,
        training_complete=True,
        optimizer=optimizer,
        best_epoch=best_epoch,
        current_lr=current_lr,
        lr_reductions=lr_reductions,
        rng=rng,
        torch=torch,
        latest_state=latest_state,
    )
    if str(epoch_checkpoint_path or '').strip() != '':
        save_torch_payload(path=epoch_checkpoint_path, payload=payload)
    return payload


def _get_runtime_targetp2_module(localization_model, device_text='cpu'):
    torch, nn = require_torch()
    _limit_torch_threads(torch)
    if '_runtime_model_cache' not in localization_model:
        localization_model['_runtime_model_cache'] = dict()
    cache = localization_model['_runtime_model_cache']
    resolved_device = resolve_torch_device(device_text=device_text)
    cache_key = str(resolved_device)
    if cache_key in cache:
        return torch, cache[cache_key], resolved_device
    config = dict(TARGETP_TORCH_DEFAULTS)
    model_config = localization_model.get('config', {})
    config.update(model_config)
    state_impl = _rnn_impl_from_state_dict(localization_model.get('state_dict', {}))
    if state_impl != '':
        config['rnn_impl'] = state_impl
    module = _build_targetp2_torch_module(
        torch=torch,
        nn=nn,
        seq_len=int(config['seq_len']),
        n_input=int(config['n_input']),
        hidden_rnn=int(config['hidden_rnn']),
        n_filters=int(config['n_filters']),
        hidden_fc=int(config['hidden_fc']),
        filter_size=int(config['filter_size']),
        n_attention=int(config['n_attention']),
        attention_size=int(config['attention_size']),
        input_keep_prob=float(config.get('input_keep_prob', 1.0)),
        encoder_keep_prob=float(config.get('encoder_keep_prob', 1.0)),
        rnn_impl=str(config.get('rnn_impl', 'torch_lstm')),
    )
    module.load_state_dict(localization_model['state_dict'], strict=True)
    module.eval()
    module.to(resolved_device)
    cache[cache_key] = module
    return torch, module, resolved_device


def predict_targetp2_torch_encoded(x, lengths, org, localization_model, device='cpu', batch_size=512):
    torch, module, resolved_device = _get_runtime_targetp2_module(
        localization_model=localization_model,
        device_text=device,
    )
    return _predict_encoded_with_module(
        torch=torch,
        module=module,
        device=resolved_device,
        x=np.asarray(x, dtype=np.float32),
        lengths=np.asarray(lengths, dtype=np.int64),
        org=np.asarray(org, dtype=np.int64),
        batch_size=int(batch_size),
    )


def predict_targetp2_torch_batch(aa_sequences, organism_groups, localization_model, device='cpu', batch_size=512):
    config = dict(TARGETP_TORCH_DEFAULTS)
    config.update(localization_model.get('config', {}))
    x, lengths = encode_targetp_blosum_sequences(
        aa_sequences=aa_sequences,
        seq_len=int(config['seq_len']),
    )
    org = np.asarray([
        organism_group_to_targetp_org(group)
        for group in organism_groups
    ], dtype=np.int64)
    return predict_targetp2_torch_encoded(
        x=x,
        lengths=lengths,
        org=org,
        localization_model=localization_model,
        device=device,
        batch_size=batch_size,
    )


def predict_targetp2_torch_localization(aa_seq, localization_model, organism_group=''):
    prob = predict_targetp2_torch_batch(
        aa_sequences=[aa_seq],
        organism_groups=[organism_group],
        localization_model=localization_model,
        device='cpu',
        batch_size=1,
    )[0]
    probs = {
        LOCALIZATION_CLASSES[i]: float(prob[i])
        for i in range(len(LOCALIZATION_CLASSES))
    }
    pred_idx = int(np.argmax(prob))
    return LOCALIZATION_CLASSES[pred_idx], probs


def export_targetp2_torch_localize_model(model_payload, training_tsv='', class_thresholds=None):
    thresholds = class_thresholds
    if thresholds is None:
        thresholds = {class_name: 1.0 for class_name in LOCALIZATION_CLASSES}
    return {
        'model_type': TARGETP_TORCH_MODEL_TYPE,
        'feature_names': list(FEATURE_NAMES),
        'localization_model': {
            'mode': 'targetp_torch',
            'class_order': list(LOCALIZATION_CLASSES),
            'config': dict(model_payload['config']),
            'state_dict': dict(model_payload['state_dict']),
            'class_thresholds': dict(thresholds),
        },
        'perox_model': {
            'mode': 'constant',
            'yes_probability': 0.0,
        },
        'metadata': {
            'training_tsv': str(training_tsv),
            'model_arch': TARGETP_TORCH_MODEL_TYPE,
            'seed': int(model_payload.get('seed', 0)),
        },
    }


def load_targetp_npz(path):
    data = np.load(path)
    return {
        'x': np.asarray(data['x'], dtype=np.float32),
        'y_cs': np.asarray(data['y_cs'], dtype=np.int64),
        'y_type': np.asarray(data['y_type'], dtype=np.int64),
        'len_seq': np.asarray(data['len_seq'], dtype=np.int64),
        'org': np.asarray(data['org'], dtype=np.int64),
        'fold': np.asarray(data['fold'], dtype=np.int64),
        'ids': np.asarray(data['ids']).astype(str),
    }


def save_torch_payload(path, payload):
    torch, _ = require_torch()
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, path)


def load_torch_payload(path):
    torch, _ = require_torch()
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _safe_fold_name(outer_fold, val_fold):
    return 'outer{}_val{}'.format(int(outer_fold), int(val_fold))


def _parse_fold_subset(value, available):
    available = [int(v) for v in available]
    text = str(value or 'all').strip().lower()
    if text in ['', 'all', '*']:
        return list(available)
    out = list()
    for part in text.split(','):
        part = part.strip()
        if part == '':
            continue
        fold_id = int(part)
        if fold_id not in available:
            raise ValueError('Fold {} is not available. Available folds: {}'.format(fold_id, available))
        out.append(fold_id)
    return out


def run_targetp2_torch_nested_oof(
    targetp_npz,
    model_dir,
    outer_folds='all',
    val_folds='all',
    reuse_cache=True,
    max_models=0,
    seed_offset=0,
    device='auto',
    **train_kwargs
):
    data = load_targetp_npz(path=targetp_npz)
    folds = data['fold']
    available = sorted(np.unique(folds).astype(int).tolist())
    selected_outer = _parse_fold_subset(outer_folds, available)
    selected_val_all = _parse_fold_subset(val_folds, available)
    os.makedirs(model_dir, exist_ok=True)

    prob_sum = np.zeros((data['x'].shape[0], len(LOCALIZATION_CLASSES)), dtype=np.float64)
    prob_count = np.zeros((data['x'].shape[0],), dtype=np.int64)
    model_rows = list()
    num_trained_or_loaded = 0

    for outer_fold in selected_outer:
        test_mask = folds == int(outer_fold)
        inner_candidates = [fold for fold in available if int(fold) != int(outer_fold)]
        selected_val = [fold for fold in selected_val_all if int(fold) in inner_candidates]
        for val_fold in selected_val:
            if int(max_models) > 0 and num_trained_or_loaded >= int(max_models):
                break
            train_mask = np.isin(folds, [fold for fold in inner_candidates if int(fold) != int(val_fold)])
            val_mask = folds == int(val_fold)
            fold_name = _safe_fold_name(outer_fold=outer_fold, val_fold=val_fold)
            checkpoint = os.path.join(model_dir, fold_name + '.pt')
            resumed_from_checkpoint = False
            if bool(reuse_cache) and os.path.exists(checkpoint):
                payload = load_torch_payload(path=checkpoint)
                requested_epochs = int(train_kwargs.get('epochs', TARGETP_TORCH_DEFAULTS['epochs']))
                cached_epochs = len(payload.get('history', []))
                if bool(payload.get('training_complete', False)) and cached_epochs >= requested_epochs:
                    used_cache = True
                else:
                    if bool(train_kwargs.get('verbose', False)):
                        print('resuming {}'.format(fold_name), flush=True)
                    payload = fit_targetp2_torch_model(
                        x_train=data['x'][train_mask],
                        y_type_train=data['y_type'][train_mask],
                        y_cs_train=data['y_cs'][train_mask],
                        len_train=data['len_seq'][train_mask],
                        org_train=data['org'][train_mask],
                        x_val=data['x'][val_mask],
                        y_type_val=data['y_type'][val_mask],
                        y_cs_val=data['y_cs'][val_mask],
                        len_val=data['len_seq'][val_mask],
                        org_val=data['org'][val_mask],
                        seed=int(seed_offset) + int(val_fold),
                        device=device,
                        resume_payload=payload,
                        epoch_checkpoint_path=checkpoint,
                        **train_kwargs
                    )
                    used_cache = False
                    resumed_from_checkpoint = True
            else:
                if bool(train_kwargs.get('verbose', False)):
                    print('training {}'.format(fold_name), flush=True)
                payload = fit_targetp2_torch_model(
                    x_train=data['x'][train_mask],
                    y_type_train=data['y_type'][train_mask],
                    y_cs_train=data['y_cs'][train_mask],
                    len_train=data['len_seq'][train_mask],
                    org_train=data['org'][train_mask],
                    x_val=data['x'][val_mask],
                    y_type_val=data['y_type'][val_mask],
                    y_cs_val=data['y_cs'][val_mask],
                    len_val=data['len_seq'][val_mask],
                    org_val=data['org'][val_mask],
                    seed=int(seed_offset) + int(val_fold),
                    device=device,
                    epoch_checkpoint_path=checkpoint,
                    **train_kwargs
                )
                save_torch_payload(path=checkpoint, payload=payload)
                used_cache = False
            localize_model = export_targetp2_torch_localize_model(model_payload=payload)
            prob = predict_targetp2_torch_encoded(
                x=data['x'][test_mask],
                lengths=data['len_seq'][test_mask],
                org=data['org'][test_mask],
                localization_model=localize_model['localization_model'],
                device=device,
                batch_size=int(train_kwargs.get('batch_size', TARGETP_TORCH_DEFAULTS['batch_size'])),
            )
            prob_sum[test_mask, :] += prob.astype(np.float64)
            prob_count[test_mask] += 1
            model_rows.append({
                'outer_fold': int(outer_fold),
                'val_fold': int(val_fold),
                'checkpoint': checkpoint,
                'used_cache': bool(used_cache),
                'n_train': int(np.sum(train_mask)),
                'n_val': int(np.sum(val_mask)),
                'n_test': int(np.sum(test_mask)),
                'resumed_from_checkpoint': bool(resumed_from_checkpoint),
                'best_metrics': payload.get('best_metrics', {}),
                'final_val_metrics': payload.get('final_val_metrics', {}),
            })
            num_trained_or_loaded += 1
        if int(max_models) > 0 and num_trained_or_loaded >= int(max_models):
            break

    covered = prob_count > 0
    prob_matrix = np.zeros_like(prob_sum, dtype=np.float64)
    prob_matrix[covered, :] = prob_sum[covered, :] / prob_count[covered].reshape((-1, 1))
    if np.any(~covered):
        prob_matrix[~covered, 0] = 1.0
    pred = np.argmax(prob_matrix[covered, :], axis=1).astype(np.int64)
    metrics = {
        'covered_rows': int(np.sum(covered)),
        'total_rows': int(data['x'].shape[0]),
        'macro_f1': _macro_f1_from_indices(data['y_type'][covered], pred, len(LOCALIZATION_CLASSES)) if np.any(covered) else 0.0,
        'overall_accuracy': float(np.mean(pred == data['y_type'][covered])) if np.any(covered) else 0.0,
    }
    return {
        'prob_matrix': prob_matrix,
        'true_idx': data['y_type'],
        'class_names': list(LOCALIZATION_CLASSES),
        'fold_ids': data['fold'],
        'covered_mask': covered,
        'prob_count': prob_count,
        'metrics': metrics,
        'models': model_rows,
    }


def write_targetp2_torch_oof_npz(path, result):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        path,
        prob_matrix=np.asarray(result['prob_matrix'], dtype=np.float64),
        true_idx=np.asarray(result['true_idx'], dtype=np.int64),
        class_names=np.asarray(result['class_names']),
        fold_ids=np.asarray(result['fold_ids'], dtype=np.int64),
        covered_mask=np.asarray(result['covered_mask'], dtype=bool),
        prob_count=np.asarray(result['prob_count'], dtype=np.int64),
    )


def write_targetp2_torch_report(path, result, profile):
    out_dir = os.path.dirname(path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    out = {
        'profile': profile,
        'metrics': result['metrics'],
        'models': result['models'],
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(out, handle, indent=2, sort_keys=True)


def class_probabilities_from_vector(prob_vec):
    prob_vec = np.asarray(prob_vec, dtype=np.float64)
    prob_vec = np.clip(prob_vec, 0.0, None)
    if float(np.sum(prob_vec)) <= 0.0:
        prob_vec = softmax(np.zeros((len(LOCALIZATION_CLASSES),), dtype=np.float64))
    else:
        prob_vec = prob_vec / float(np.sum(prob_vec))
    return {
        LOCALIZATION_CLASSES[i]: float(prob_vec[i])
        for i in range(len(LOCALIZATION_CLASSES))
    }
