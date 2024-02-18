# %%
import random
from .utils import total_loglikelihood, tokens_loglikelihoods, encode_text, decode_tokens
from typing import List, Tuple, Dict
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPACES_MAP = [
    "\u2000",
    "\u2002",
    "\u2005\u200A\u2006",
    "\u2006\u2006\u2006",
    "\u2007",
    "\u202f\u2006\u200A",
    "\u205f\u2006\u2006",
]
from silver_speak.identical_map import chars_map

# SPACES_MAP = [
#    "\u2007\u2062",
# ]


def replace_spaces(text):
    # Replaces all spaces in text with a random space from the SPACES_MAP
    perturbed_text = ""
    words = text.split(" ")
    for word in words:
        perturbed_text += word + random.choice(SPACES_MAP)
    return perturbed_text[:-1]  # Remove last space


def convert_to_char_from_hex(hex_num):
    # append 0x to hex num string to convert it.
    hex_num = "0x" + hex_num.strip(" ")
    # get the actual integer of the specific hex num with base 16.
    hex_num = int(hex_num, base=16)
    # finally get the actual character stored for specific hex char representation.
    hex_num = chr(hex_num)
    return hex_num


from silver_speak.utils import (
    encode_text,
    tokens_loglikelihoods,
    replace_characters,
    decode_tokens,
)


def decrease_loglikelihood_replace_characters_by_equivalents(
    chars_map, text, patience=10
):
    encoded_text = encode_text(text)
    loglikelihoods = tokens_loglikelihoods(encoded_text)
    print(
        f"Mean starting loglikelihood: {sum([x[1] for x in loglikelihoods]) / len(loglikelihoods)}"
    )
    current_loglikelihood = sum([x[1] for x in loglikelihoods]) / len(loglikelihoods)
    global_best_loglikelihood = current_loglikelihood
    global_best_text = encoded_text.tolist()
    current_used_patience = 0
    try:
        while patience > current_used_patience:
            new_tokens_list = replace_characters(
                chars_map, loglikelihoods, num_to_replace=1
            )
            loglikelihoods = tokens_loglikelihoods(new_tokens_list)
            current_loglikelihood = sum([x[1] for x in loglikelihoods]) / len(
                loglikelihoods
            )
            print(f"Mean loglikelihood: {current_loglikelihood}")
            print(f"New text: {decode_tokens(new_tokens_list)}")
            if current_loglikelihood < global_best_loglikelihood:
                global_best_loglikelihood = current_loglikelihood
                global_best_text = new_tokens_list.tolist()
                current_used_patience = 0
            else:
                current_used_patience += 1
    except ValueError:
        print("No more characters to replace.")

    # Reconstruct the text
    text = decode_tokens(global_best_text)
    return text


def replace_characters_by_equivalents(final_map, text):
    """This is an attack where we replace only the negative sentiment words found in negative-words list."""

    # Replace all chars in text with a random char from the final_map
    rewritten_text = ""
    rewrite = True
    for word in text.split(" "):
        if random.random() < 0.0:
            if random.random() < 0.4 and not rewrite:
                rewrite = not rewrite  # flip the rewrite flag
            else:
                rewrite = not rewrite  # flip the rewrite flag
        if not rewrite:
            rewritten_text += word + " "
            continue
        for char in word:
            if char in final_map.keys():
                rewritten_text += random.choice(final_map[char])
            else:
                # other type of character so write it to file as it is.
                rewritten_text += char
        rewritten_text += " "

    return rewritten_text


class TreeNode:
    parent = None
    changed_index: int
    changed_letter: str
    loglikelihood: float = None

    def get_text(self) -> str:
        # Go to the parent and add the change of this node
        if self.parent:
            original_text = self.parent.get_text()
            changed_text = original_text[: self.changed_index] + self.changed_letter + original_text[self.changed_index + 1 :]
            return changed_text
        else:
            # We are the root
            raise NotImplementedError("This is not a root node, but does not have a parent")

    def get_loglikelihood(self) -> float:
        if self.loglikelihood:
            return self.loglikelihood
        loglikelihoods = tokens_loglikelihoods(encode_text(self.get_text()))
        self.loglikelihood = total_loglikelihood(loglikelihoods)
        return self.loglikelihood

    def dump_to_file(self, filename: str = 'current_node.txt'):
        with open(filename, 'w') as f:
            f.write('Dump of the tree node\n')
            ascendants = []
            current_node = self
            while current_node:
                ascendants.append(current_node)
                current_node = current_node.parent
            for i, node in enumerate(reversed(ascendants)):
                f.write(f"Level {i}: {node}\n")
                f.write(f"Text: {node.get_text()}\n")
                f.write(f"Loglikelihood: {node.get_loglikelihood()}\n")
                f.write("\n")
    
    def __repr__(self) -> str:
        return f"TreeNode(change=[{self.changed_index}] -> {self.changed_letter})"
    
class RootTreeNode(TreeNode):
    text: str

    def get_text(self) -> str:
        return self.text
    
    def __repr__(self) -> str:
        return f"RootTreeNode()"


def generate_child_nodes(node: TreeNode):
    """
    Given a certain text, this function generates all possible modifications of the given text at that point.
    """
    children = []
    replaceable_letters = set(chars_map.keys())
    exploded_text = list(node.get_text())
    logger.info("Generating children")
    for i, letter in enumerate(exploded_text):
        if letter in replaceable_letters:
            for new_letter in chars_map[letter]:
                new_child = TreeNode()
                new_child.parent = node
                new_child.changed_index = i
                new_child.changed_letter = new_letter
                children.append(new_child)
                logger.debug(f"New child: {new_child}")
    logger.info(f"Generated {len(children)} children")
    return children


# Ideas to improve performance:
# Bounds: estimate the possible likelihood change depending on the position of the change
# Bounds: estimate the possible likelihood change depending on the previous improvements already computed
# Do Bayesian optimization to estimate the best changes to make - this means exploring the space of possibilities (i.e. the positions in the text) in a more intelligent way


def rewrite_attack(
    text, replace_chars_fn=replace_characters_by_equivalents, do_replace_spaces=True
):
    """
    Branch and bound algorithm.

    The goal of this function is to explore the space of possibilities to reach the optimal solution. There is a tree of possibilities, where each node is one instance of the text with a certain set of modifications applied.

    The optimal solution is such that loglikelihood is minimal.

    The space of possibilities at a given node in the tree of possibilities is given by all 1-letter replacements possible at that version of the text.
    """
    THRESHOLD = 0.5

    current_node = RootTreeNode()
    current_node.text = text

    # Branch and bound algorithm
    best_node = current_node
    best_loglikelihood = best_node.get_loglikelihood()
    current_nodes = [current_node]
    improvement = 1e100 # Improvement is the ratio of the current loglikelihood to the best loglikelihood
    best_improvement = 0
    while current_nodes and improvement > (THRESHOLD * best_improvement):
        # Stop if we haven't improved in a while - this is that the improvement is less than 50% of the best improvement
        # Sort the nodes by loglikelihood
        logger.info("Sorting nodes")
        current_nodes.sort(key=lambda x: x.get_loglikelihood())
        # Remove all nodes after the 50th
        current_nodes = current_nodes[:50]
        current_node = current_nodes.pop(0)
        logger.info(f"Current loglikelihood: {current_node.get_loglikelihood()}")
        if current_node.get_loglikelihood() < best_loglikelihood:
            improvement = current_node.get_loglikelihood() / best_loglikelihood
            if improvement > best_improvement:
                best_improvement = improvement
            best_node = current_node
            best_loglikelihood = best_node.get_loglikelihood()
            logger.info(f"New best loglikelihood: {best_loglikelihood}, improvement: {improvement}, best improvement: {best_improvement}")
            logger.info(f"Best node: {best_node}")
            best_node.dump_to_file()
        children = generate_child_nodes(current_node)
        current_nodes += children

    # Log the best node's tree
    logger.info("Best node's tree:")
    current_node = best_node
    while current_node:
        logger.info(current_node)
        current_node = current_node.parent
    return best_node.get_text()
