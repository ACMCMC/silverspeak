# %%
"""
We'll experiment with the perplexity of the models w.r.t. the text in this file.
"""
test_text = """What are the standards required of offered properties? Properties need to be habitable and must meet certain health and safety standards, which the local authority can discuss with you. These standards have been agreed by the Department of Housing, Local Government and Heritage. The local authority will assess your property to make sure it meets the standards. If the property does not meet the standards, the local authority will explain why and can discuss what could be done to bring the property up to standard. Some properties may not be suitable for all those in need of accommodation, due to location or other reasons. However, every effort will be made by the local authority to ensure that offered properties are matched to appropriate beneficiaries."""

# Get the loglikelihoods of the tokens
from silver_speak.utils import add_fill_tokens, align_two_token_sequences, get_loglikelihoods_of_tokens, total_loglikelihood, encode_text, get_filled_ranges
import torch

input_ids = encode_text(test_text)
original_toks_loglikelihoods = get_loglikelihoods_of_tokens(input_ids=input_ids) # List of (tok_id, loglikelihood) tuples
original_toks = torch.tensor([tok_id for tok_id, loglikelihood in original_toks_loglikelihoods])
original_loglikelihoods = torch.tensor([loglikelihood for tok_id, loglikelihood in original_toks_loglikelihoods])

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
from silver_speak.optimized_attack import optimized_attack
# Change one of the characters and plot the loglikelihoods again
changed_text = optimized_attack(test_text, percentage_to_replace=0.05, random_seed=42)
print(changed_text)

changed_input_ids = encode_text(changed_text)
changed_toks_loglikelihoods = get_loglikelihoods_of_tokens(input_ids=changed_input_ids)
changed_toks = torch.tensor([tok_id for tok_id, loglikelihood in changed_toks_loglikelihoods])
changed_loglikelihoods = torch.tensor([loglikelihood for tok_id, loglikelihood in changed_toks_loglikelihoods])
aligned_toks = align_two_token_sequences(reference=changed_toks, target=original_toks)
aligned_loglikelihoods = add_fill_tokens(reference=aligned_toks, target=original_loglikelihoods, ELEMENT_TO_FILL=torch.nan) # This makes the plot not show the fill tokens
# %%
# Plot the original loglikelihoods w.r.t. the changed text in the same plot, to compare, with different colors
plt.figure(figsize=(10, 5))

# Plot the loglikelihoods
plt.plot(aligned_loglikelihoods, label='Original text')
plt.plot(changed_loglikelihoods, label='Changed text', alpha=0.5)

# Draw a gray overlay to indicate the fill tokens, between x=40 and x=50. Fill the entire y-axis. Do not draw an outline
for start, end in get_filled_ranges(aligned_toks):
    # Use avxspan to fill the area between the two lines
    plt.axvspan(start - 1, end + 1, color='gray', alpha=0.2, edgecolor=None, linewidth=0, hatch='//')
# Add it to the legend
plt.fill_between([], [], color='gray', alpha=0.2, label='Fill tokens', hatch='//')

plt.xlabel('Token index')
plt.ylabel('Loglikelihood')
plt.title('Loglikelihoods of the tokens in the text')
plt.legend()
plt.show()
# %%