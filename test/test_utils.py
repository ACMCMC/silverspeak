# %%
import pytest
import torch

from silver_speak.utils import encode_text, align_two_token_sequences, add_fill_tokens


def test_align_two_token_sequences():
    reference = torch.Tensor([15496, 0, 347, 14363, 318, 257, 1332, 13])
    target =    torch.Tensor([15496, 0, 770,        318, 257, 1332, 13])

    aligned = align_two_token_sequences(reference, target)

    assert aligned.tolist() == [15496, 0, 770, -1, 318, 257, 1332, 13]

def test_align_two_token_sequences_2():
    reference = torch.Tensor([15496, 0, 347, 14363, 257, 1332, 13])
    target =    torch.Tensor([15496, 0, 770,   318, 257, 1332, 13])

    aligned = align_two_token_sequences(reference, target)

    assert aligned.tolist() == [15496, 0, 770,   -1, 257, 1332, 13]

def test_add_fill_tokens():
    reference = torch.Tensor([15496, 0, -1, 318, -1, 1332, 13])
    target =    torch.Tensor([1,     2,  3,   4,  5])

    aligned = add_fill_tokens(reference, target)

    assert aligned.tolist() == [1, 2, -1, 3, -1, 4, 5]
# %%
