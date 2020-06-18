import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence

PADDING_TOKEN = 0


def get_device(model):
  return next(model.parameters()).get_device()


def trim(sequence):
  if sequence.dim() == 1:
    return sequence[sequence != PADDING_TOKEN]
  else:
    shape = sequence.shape
    sequence = sequence.reshape(-1, shape[-1])
    sequences = []
    for s in sequence:
      sequences.append(trim(s))
    ret = pad_sequence(sequences, batch_first=True,
                       padding_value=PADDING_TOKEN)

    return ret.reshape(tuple(shape[:-1]) + (-1,))


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
  try:
    device = ids.get_device()
    ids = ids.cpu().numpy()
  except AttributeError:
    device = None
  dim = ids.ndim
  shape = ids.shape
  eos_tensor = np.array([eos])
  if dim == 3:
    ids = ids.reshape(-1, shape[-1])
  id_list = []
  for id in ids:
    eos_index = np.where(id == eos)[0]
    sequences = []
    start_index = 0
    for i in eos_index:
      if i == eos_index[0]:
        end_index = i
      else:
        end_index = i + 1
      sequences.append(id[start_index:end_index])
      start_index = i + 1
    last_id = id[start_index:]
    last_id = last_id[last_id != PADDING_TOKEN]
    if len(last_id) == 0:
      sequences.insert(0, eos_tensor)
    else:
      sequences.append(eos_tensor)
      sequences.append(last_id)

    sequence = np.concatenate(tuple(reversed(sequences)))
    if device is not None:
      sequence = torch.tensor(sequence)
    id_list.append(sequence)

  if device is None:
    return id_list
  else:
    ret = pad_sequence(id_list, batch_first=True, padding_value=PADDING_TOKEN)

    if dim == 3:
      ret = ret.reshape(shape[0], shape[1], -1)

    ret = ret.to(device)

    return ret


# def reverse(ids, eos):
#   dim = ids.dim()
#   shape = ids.shape
#   if dim == 3:
#     ids = ids.reshape(-1, shape[-1])
#   eos_tensor = torch.tensor([eos]).to(ids.get_device())
#   id_list = []
#   for id in ids:
#     eos_index = (id == eos).nonzero()
#     sequences = []
#     start_index = 0
#     for i in eos_index:
#       if i == eos_index[0]:
#         end_index = i
#       else:
#         end_index = i + 1
#       sequences.append(id[start_index:end_index])
#       start_index = i + 1
#     last_id = trim(id[start_index:])
#     if len(last_id) == 0:
#       sequences.insert(0, eos_tensor)
#     else:
#       sequences.append(eos_tensor)
#       sequences.append(last_id)
#
#     id_list.append(torch.cat(tuple(reversed(sequences))))
#
#   ret = pad_sequence(id_list, batch_first=True, padding_value=PADDING_TOKEN)
#
#   if dim == 3:
#     ret = ret.reshape(shape[0], shape[1], -1)
#
#   return ret

def pad_sequence(sequences, batch_first=False, padding_value=0, pad_end=True):
  r"""Pad a list of variable length Tensors with ``padding_value``

  ``pad_sequence`` stacks a list of Tensors along a new dimension,
  and pads them to equal length. For example, if the input is list of
  sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
  otherwise.

  `B` is batch size. It is equal to the number of elements in ``sequences``.
  `T` is length of the longest sequence.
  `L` is length of the sequence.
  `*` is any number of trailing dimensions, including none.

  Example:
      >>> from torch.nn.utils.rnn import pad_sequence
      >>> a = torch.ones(25, 300)
      >>> b = torch.ones(22, 300)
      >>> c = torch.ones(15, 300)
      >>> pad_sequence([a, b, c]).size()
      torch.Size([25, 3, 300])

  Note:
      This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
      where `T` is the length of the longest sequence. This function assumes
      trailing dimensions and type of all the Tensors in sequences are same.

  Arguments:
      sequences (list[Tensor]): list of variable length sequences.
      batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
          ``T x B x *`` otherwise
      padding_value (float, optional): value for padded elements. Default: 0.

  Returns:
      Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
      Tensor of size ``B x T x *`` otherwise
  """

  # assuming trailing dimensions and type of all the Tensors
  # in sequences are same and fetching those from sequences[0]
  max_size = sequences[0].size()
  trailing_dims = max_size[1:]
  max_len = max([s.size(0) for s in sequences])
  if batch_first:
    out_dims = (len(sequences), max_len) + trailing_dims
  else:
    out_dims = (max_len, len(sequences)) + trailing_dims

  out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
  for i, tensor in enumerate(sequences):
    length = tensor.size(0)
    # use index notation to prevent duplicate references to the tensor
    if batch_first:
      if pad_end:
        out_tensor[i, :length, ...] = tensor
      else:
        out_tensor[i, max_len - length:, ...] = tensor
    else:
      if pad_end:
        out_tensor[:length, i, ...] = tensor
      else:
        out_tensor[max_len - length:, i, ...] = tensor

  return out_tensor
