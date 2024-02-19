from silver_speak.utils import (
    encode_text,
    get_loglikelihoods_of_tokens,
    total_loglikelihood,
)
import logging
from silver_speak.identical_map import chars_map
from silver_speak.silver_speak import replace_characters_by_equivalents

logger = logging.getLogger(__name__)


class TreeNode:
    parent = None
    changed_index: int
    changed_letter: str
    loglikelihood: float = None

    def get_text(self) -> str:
        # Go to the parent and add the change of this node
        if self.parent:
            original_text = self.parent.get_text()
            changed_text = (
                original_text[: self.changed_index]
                + self.changed_letter
                + original_text[self.changed_index + 1 :]
            )
            return changed_text
        else:
            # We are the root
            raise NotImplementedError(
                "This is not a root node, but does not have a parent"
            )

    def get_loglikelihood(self) -> float:
        if self.loglikelihood:
            return self.loglikelihood
        loglikelihoods = get_loglikelihoods_of_tokens(encode_text(self.get_text()))
        self.loglikelihood = total_loglikelihood(loglikelihoods)
        return self.loglikelihood

    def dump_to_file(self, filename: str = "current_node.txt"):
        with open(filename, "w") as f:
            f.write("Dump of the tree node\n")
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


def star_rewrite_attack(
    text, replace_chars_fn=replace_characters_by_equivalents, do_replace_spaces=True
):
    """
    SilverSpeak* attach - best attack, but slowest.

    Branch and bound algorithm.

    The goal of this function is to explore the space of possibilities to reach the optimal solution. There is a tree of possibilities, where each node is one instance of the text with a certain set of modifications applied.

    The optimal solution is such that loglikelihood is minimal.

    The space of possibilities at a given node in the tree of possibilities is given by all 1-letter replacements possible at that version of the text.
    """
    THRESHOLD = 0.3

    current_node = RootTreeNode()
    current_node.text = text

    # Branch and bound algorithm
    best_node = current_node
    best_loglikelihood = best_node.get_loglikelihood()
    current_nodes = [current_node]
    improvement = 1e100  # Improvement is the ratio of the current loglikelihood to the best loglikelihood
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
            logger.info(
                f"New best loglikelihood: {best_loglikelihood}, improvement: {improvement}, best improvement: {best_improvement}"
            )
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
