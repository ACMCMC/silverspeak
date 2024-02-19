# %%
"""
We'll experiment with the perplexity of the models w.r.t. the text in this file.
"""
test_text = """What are the standards required of offered properties? Properties need to be habitable and must meet certain health and safety standards, which the local authority can discuss with you. These standards have been agreed by the Department of Housing, Local Government and Heritage. The local authority will assess your property to make sure it meets the standards. If the property does not meet the standards, the local authority will explain why and can discuss what could be done to bring the property up to standard. Some properties may not be suitable for all those in need of accommodation, due to location or other reasons. However, every effort will be made by the local authority to ensure that offered properties are matched to appropriate beneficiaries."""

# Get the loglikelihoods of the tokens
from silver_speak.utils import add_fill_tokens, align_two_token_sequences, get_loglikelihoods_of_tokens, total_loglikelihood, encode_text
import torch

input_ids = encode_text(test_text)
original_toks_loglikelihoods = get_loglikelihoods_of_tokens(input_ids=input_ids) # List of (tok_id, loglikelihood) tuples
original_toks = torch.tensor([tok_id for tok_id, loglikelihood in original_toks_loglikelihoods])
original_loglikelihoods = torch.tensor([loglikelihood for tok_id, loglikelihood in original_toks_loglikelihoods])
# %%
# Plot the loglikelihoods
import matplotlib.pyplot as plt
import numpy as np
# Label the tokens
plt.figure(figsize=(10, 5))
plt.plot(original_loglikelihoods)
plt.xlabel('Token index')
plt.ylabel('Loglikelihood')
plt.title('Loglikelihoods of the tokens in the text')
plt.show()

# %%
# Change one of the characters and plot the loglikelihoods again
changed_text = """What are the standards required of offered properties? Properties need to be habitable and must meet certain health and safety standards, which the local authority can discuss with you. 𐤯hese standards have been agreed by the Department of Housing, Local Goνernment and Heritage. The local authority will assess your property to make sսre it meets the standards. If the property does not meet the standards, the local authority will explain why and can discuss what could be done to brinɡ the property up to standard. Some properties may not be suitable for all those in need of accommodation, due to location or other reasons. However, every effort will be made by the local authority to ensure that offered properties are matched to appropriate beneficiaries."""
changed_input_ids = encode_text(changed_text)
changed_toks_loglikelihoods = get_loglikelihoods_of_tokens(input_ids=changed_input_ids)
changed_toks = torch.tensor([tok_id for tok_id, loglikelihood in changed_toks_loglikelihoods])
changed_loglikelihoods = torch.tensor([loglikelihood for tok_id, loglikelihood in changed_toks_loglikelihoods])
aligned_toks = align_two_token_sequences(reference=changed_toks, target=original_toks)
aligned_loglikelihoods = add_fill_tokens(reference=aligned_toks, target=original_loglikelihoods)
# %%
# Plot the original loglikelihoods w.r.t. the changed text in the same plot, to compare, with different colors
plt.figure(figsize=(10, 5))
plt.plot(aligned_loglikelihoods, label='Original text')
plt.plot(changed_loglikelihoods, label='Changed text')
plt.xlabel('Token index')
plt.ylabel('Loglikelihood')
plt.title('Loglikelihoods of the tokens in the text')
plt.legend()
plt.show()
# %%