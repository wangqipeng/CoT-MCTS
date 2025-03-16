import math

PRM_THRESHOLD = 0.75

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
    
def uct(node):
    if node.visits == 0:
        return float('inf')
    parent_visits = node.parent.visits if node.parent else 1
    return node.value / node.visits + 1.414 * (math.log(parent_visits) / node.visits) ** 0.5

def generate_steps(current_state, model):
    prompt = f"Given '{current_state}', suggest 2-3 possible next steps."
    response = model.generate(prompt)
    return response.split('\n')

def simulate(state, model, prm):
    prompt = f"Starting from '{state}', solve step by step and verify"
    cot = model.generate(prompt)
    steps = cot.split('\n')
    score = 1.0
    for step in steps:
        prm_score = prm_predict(step, prm)
        if prm_score < PRM_THRESHOLD: #wrong step
            score = 0
            break
        score *= prm_score
    return score, step[-1] if steps else state

def mcts_cot(problem, model, prm, iter_num = 100):
    root = Node(problem)
    for _ in range(iter_num):
        # selection
        node = root
        while node.children and all(c.visits > 0 for c in node.children):
            node = max(node.children, key = uct)

        # expansion
        if node.visits > 0:
            next_steps = generate_steps(node.state, model)
            for step in next_states:
                node.children.append(Node(step, node))
            node = node.children[0]

        #simulate
        value, final_state = simulate(node.state, model, prm)

        #backpropagation
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    hidden_cot = extract_best_path(root)
    return hidden_cot

def extract_best_path(root):
    path = []
    node = root
    while node.children:
        node = max(node.children, key=lambda n : n.value / n.visits if n.visits > 0 else 0)
        path.append(node.state)
    return path

def prm_predict(step, prm):
    prompt = f"Evaluate this step: '{step}'. Return probability of correctness (0-1)."
    return float(prm.generate(prompt))