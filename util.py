import torch

from torch.nn.utils.rnn import pad_sequence

PADDING_TOKEN = 0


def get_device(model):
  return next(model.parameters()).get_device()


def trim(sequence):
  return sequence[sequence != PADDING_TOKEN]
  # pad_index = (sequence == PADDING_TOKEN).nonzero()
  # if sequence.dim() == 1:
  #   if len(pad_index) == 0:
  #     end_index = len(sequence)
  #   else:
  #     end_index = pad_index[0].item()
  #   return sequence[:end_index]
  # else:
  #   if len(pad_index) == 0:
  #     end_index = sequence.shape[1]
  #   else:
  #     end_index = pad_index[0][1].item()
  #   return sequence[:, :end_index]


def concat(a, b, eos=None):
  if eos is not None:
    eos = torch.tensor([eos]).to(a.get_device())

  if b.dim() == 2:
    b = b.unsqueeze(1)

  sequences = []
  for i in range(len(a)):
    a_i = trim(a[i])
    b_is = []
    for j in range(b.shape[1]):
      b_is.append(trim(b[i, j]))

    for b_i in b_is:
      if eos is None:
        sequences.append(torch.cat((a_i, b_i)))
      else:
        sequences.append(torch.cat((a_i, eos, b_i)))

  ret = pad_sequence(sequences, batch_first=True, padding_value=PADDING_TOKEN)

  if b.shape[1] != 1:
    ret = ret.reshape(b.shape[0], b.shape[1], -1)

  return ret


def reverse(ids, eos):
  dim = ids.dim()
  shape = ids.shape
  if dim == 3:
    ids = ids.reshape(-1, shape[-1])
  eos_tensor = torch.tensor([eos]).to(ids.get_device())
  id_list = []
  for id in ids:
    eos_index = (id == eos).nonzero()
    sequences = []
    start_index = 0
    for i in eos_index:
      if i == eos_index[0]:
        end_index = i
      else:
        end_index = i + 1
      sequences.append(id[start_index:end_index])
      start_index = i + 1
    last_id = trim(id[start_index:])
    if len(last_id) == 0:
      sequences.insert(0, eos_tensor)
    else:
      sequences.append(eos_tensor)
      sequences.append(last_id)

    id_list.append(torch.cat(tuple(reversed(sequences))))

  ret = pad_sequence(id_list, batch_first=True, padding_value=PADDING_TOKEN)

  if dim == 3:
    ret = ret.reshape(shape[0], shape[1], -1)

  return ret
