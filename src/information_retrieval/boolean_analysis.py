from information_retrieval.indexer import Indexer
from information_retrieval.normalizer import Normalizer


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.size() == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        try:
            return self.items[len(self.items) - 1]
        except KeyError:
            return None

    def size(self):
        return len(self.items)

    def stack_pop_order(self):
        return list(reversed(self.items))

    def reversed_stack_pop_order(self):
        return self.items

    def reverse(self):
        result = Stack()
        for value in reversed(self.items):
            result.push(value)
        return result


class Node:
    def __init__(self, value):
        self._parent = None
        self._children = []
        self.value = value

    def add_child(self, node):
        self._children.append(node)
        node._parent = self

    def get_parent_node(self):
        return self._parent

    def __str__(self):
        if len(self._children) == 0:
            return self.value.__str__()

        if self.value == "not":
            return "not " + self._children[0].__str__()

        result = ""
        if self._parent is not None:
            result += "("
        for i, child in enumerate(self._children):
            result += child.__str__()
            if i < len(self._children) - 1:
                result += " " + self.value + " "
        if self._parent is not None:
            result += ")"
        return result


non_normalize_terms = {"(", ")", "and", "or", "not"}


def create_parse_tree(query, indexer):
    # Break the query into the components. Do not use punctuation removal here.
    lower_case_query = query.lower()

    # We have to keep in mind that the query might contain brackets without accompanying spaces.
    spaced_query = lower_case_query.replace("(", " ( ").replace(")", " ) ")
    query_tokens = spaced_query.split()

    # We still have to normalize the terms in the query.
    for i, term in enumerate(query_tokens):
        if not non_normalize_terms.__contains__(query_tokens[i]):
            query_tokens[i] = indexer.normalizer.normalize(term)

    # Convert all the tokens to nodes, to be consistent.
    node_query_tokens = [Node(token) for token in query_tokens]

    # Now create subtrees.
    return create_parse_subtree(node_query_tokens)


def create_parse_subtree(token_nodes):
    # Keep a stack of nodes we have visited, which we will manipulate.
    visited_stack = Stack()

    # First find subsets in the tokens array that are contained by brackets, and process them as a subtree.
    process_brackets_as_subtrees(token_nodes, visited_stack)

    # First, merge not operator nodes with their values.
    processed_nodes_stack = process_not_operators(visited_stack)

    # With the not operators abstracted, we can start making subtrees of the and/or operator and their associated nodes.
    # We should put priority on making the subtrees for and groups first, as we can have multiple occurrences.
    processed_nodes_stack = process_and_operators(processed_nodes_stack)

    # Now process all or operators.
    processed_nodes_stack = process_or_operators(processed_nodes_stack)

    # We should only have one node in the stack now. Throw an exception if we have not.
    if processed_nodes_stack.size() != 1:
        raise Exception("We should not have ended up with multiple root nodes in the parsed subtree!")

    # Return the top node.
    return processed_nodes_stack.pop()


def process_brackets_as_subtrees(token_nodes, visited_stack):
    for node in token_nodes:
        # If the term is a right bracket, reverse the order.
        if node.value == ")":
            # Now we can reverse order. Pop nodes from the visited stack until we encounter a left bracket.
            sub_tree_tokens = Stack()

            try:
                # Add all the nodes between the brackets to a stack.
                while visited_stack.peek().value != "(":
                    sub_tree_tokens.push(visited_stack.pop())
            except IndexError:
                raise Exception("Unbalanced brackets.")

            # Pop the left bracket from the stack, as we won't need it.
            visited_stack.pop()

            # Reverse the sub tree tokens list.
            sub_tree_tokens = sub_tree_tokens.stack_pop_order()

            # Create a subtree from the new tokens, and add the found subtree to the visited stack.
            visited_stack.push(create_parse_subtree(sub_tree_tokens))
        else:
            # Add elements to the visited stack otherwise.
            visited_stack.push(node)


def process_not_operators(visited_stack):
    # Combine all not nodes with their successor, except for when we have two not nodes after each other.
    # We want to start at the start of the stack, so we want to iterate over the stack in reverse order.
    # I.e. we take from the bottom instead of the top of the stack.
    simplified_token_nodes = visited_stack.reverse()
    processed_nodes = Stack()
    while not simplified_token_nodes.is_empty():
        token_node = simplified_token_nodes.pop()
        if token_node.value == "not":
            # Get the next node and combine. Skip if next one is not as well.
            next_token_node = simplified_token_nodes.pop()
            if next_token_node.value == "not":
                continue

            # Make the successor node a child of the not node.
            token_node.add_child(next_token_node)

        # Add the token node to the processed nodes stack.
        processed_nodes.push(token_node)

    # Return the processed nodes as a stack, we want to start with the first element added, so reverse the order.
    return processed_nodes.reverse()


def process_and_operators(token_stack):
    # Use the template with the keyword "and", which will group up all ands and convert them to subtrees.
    return process_and_or_operators_template(token_stack, "and")


def process_or_operators(token_stack):
    # Use the template with the keyword "or", which will group up all ors and convert them to subtrees.
    return process_and_or_operators_template(token_stack, "or")


def process_and_or_operators_template(token_stack, keyword):
    # Convert the list to a stack, as it is more useful for us.

    # Iterate through all the nodes, and make and subtrees.
    # Save it as a stack, so that we can easily make adjustments when needed.
    processed_nodes = Stack()
    while not token_stack.is_empty():
        node = token_stack.pop()

        # If we encounter an and/or node, we want to merge the predecessor and successor of this node with the and/or.
        if node.value == keyword:
            # Find the predecessor. By the programs logic, it is on top of the processed nodes stack.
            predecessor = processed_nodes.pop()
            successor = token_stack.pop()

            # Check if the predecessor is already an and/or node. If it is, add the successor to this node instead.
            if predecessor.value == keyword:
                and_node = predecessor
                and_node.add_child(successor)
            else:
                # Otherwise create a new node we can add the predecessor and successor to.
                and_node = Node(keyword)
                and_node.add_child(predecessor)
                and_node.add_child(successor)
                pass

            # Restore the and/or node to the stack.
            processed_nodes.push(and_node)
        else:
            # If it is not an and node, we will just add it to the processed stack.
            processed_nodes.push(node)

    # We want to start at the start of the stack again, so reverse to order.
    return processed_nodes.reverse()


def simplify_parse_tree(root_node):
    # Here we want to simplify the parse tree. I.e. remove unneeded brackets.
    return root_node

indexer = Indexer()
indexer.normalizer = Normalizer(True, "None")
tree = create_parse_tree("help and not (warcraft and cookies and food or (partner or cheese and pepper))", indexer)
print(tree)
tree = create_parse_tree("help and not not (warcraft and cookies and food or partner or cheese and pepper)", indexer)
print(tree)
tree = create_parse_tree("help or not cookie", indexer)
print(tree)
tree = create_parse_tree("help and not cookie or cookies", indexer)
print(tree)
tree = create_parse_tree("help and not cookie", indexer)
print(tree)
