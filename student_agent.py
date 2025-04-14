# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
import random
from collections import defaultdict
import gdown


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action, generate = True):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved and generate:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


class SymmetricNTupleNetwork:
    def __init__(self, tuple_shapes = [], num_values=15):
        self.num_values = num_values
        self.tuple_shapes = tuple_shapes
        self.symmetry_tuples = self._generate_all_symmetric_tuples(tuple_shapes)
        self.luts = [defaultdict(float) for _ in tuple_shapes]  # one shared LUT per original tuple

    def encode(self, board_value):
        if board_value == 0:
            return 0
        return min(int(np.log2(board_value)), self.num_values - 1)

    def _rotate_index(self, idx, k):
        x, y = idx
        for _ in range(k):
            x, y = y, 3 - x
        return x, y

    def _mirror_index(self, idx):
        x, y = idx
        return x, 3 - y

    def _generate_all_symmetric_tuples(self, base_tuples):
        sym_tuples = []
        for tup in base_tuples:
            for k in range(4):
                rotated = [self._rotate_index(i, k) for i in tup]
                sym_tuples.append(rotated)
                mirrored = [self._mirror_index(i) for i in rotated]
                sym_tuples.append(mirrored)
        return sym_tuples

    def get_index(self, board, tuple_indices):
        # value = 0
        # for idx in tuple_indices:
        #     x, y = idx
        #     value = value * self.num_values + self.encode(board[x][y])
        # return value
        index = tuple(self.encode(board[x][y]) for x, y in tuple_indices)
        return index

    def value(self, board):
        val = 0
        for i, tup in enumerate(self.symmetry_tuples):
            lut_index = i // 8  # each original tuple has 8 symmetric variants
            index = self.get_index(board, tup)
            val += self.luts[lut_index][index]
        return val

    def update(self, board, target, alpha):
        value_before = self.value(board)
        # print("Value before update:", value_before)
        # print("Target:", target)
        td_error = target - value_before
        for i, tup in enumerate(self.symmetry_tuples):
            lut_index = i // 8
            index = self.get_index(board, tup)
            self.luts[lut_index][index] += alpha * td_error/len(self.symmetry_tuples)

    def save(self, path):
        serializable_luts = [dict(lut) for lut in self.luts]
        with open(path, 'wb') as f:
            pickle.dump((self.tuple_shapes, self.num_values, serializable_luts), f)

    def load(self, path):
        with open(path, 'rb') as f:
            print("start_load")
            self.tuple_shapes, self.num_values, saved_luts = pickle.load(f)
            print("tuple_shapes:", self.tuple_shapes)
            print("num_values:", self.num_values)
            # print("saved_luts:", saved_luts)
            self.symmetry_tuples = self._generate_all_symmetric_tuples(self.tuple_shapes)
            self.luts = [defaultdict(float, lut) for lut in saved_luts]



import os
# https://drive.google.com/file/d/1stLf9NqvkkmUTqvPe9dSREO0KfMAlN0s/view?usp=sharing
# Download the file only if not already downloaded
# file_id = "1stLf9NqvkkmUTqvPe9dSREO0KfMAlN0s"
# output_path = "ntuple_network.pkl"

# if not os.path.exists(output_path):
#     gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
#     print(f"Downloaded {output_path} from Google Drive.")

# tuple_shapes = [
#         [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],
#         [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2)],
#         [(1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (3, 1)],
#         [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2)],
#         [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)],
#         [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (1, 3)],
#         [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 2)],
#         [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3)],
#     ]
approximator = SymmetricNTupleNetwork()
print("start_log")
approximator.load("converted_weights.pkl")
    
def get_action(state, score):
    print("1")
    global approximator
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    td_mcts = TemporalMCTS(env, approximator, num_iterations=15)
    root = BaseNode(state, score, environment=env)

    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 
    
    for _ in range(td_mcts.num_iterations):
        td_mcts.execute_simulation(root)

    best_action, distribution = td_mcts.best_action_distribution(root)
    return best_action # Choose a random action

import random
import copy
import math
import numpy as np

class BaseNode:
    def __init__(self, board_state, current_score, parent_node=None, action_taken=None, environment=None):
        self.board_state = board_state
        self.current_score = current_score
        self.parent_node = parent_node
        self.action_taken = action_taken
        self.children_nodes = {}  # action -> Node
        self.visitation_count = 0
        self.accumulated_reward = 0.0
        self.available_actions = {}  # action -> (next_board_state, next_score)
        
        if environment is not None:
            for move in range(4):  # 4 possible moves (0, 1, 2, 3)
                simulated_env = copy.deepcopy(environment)
                simulated_env.board = board_state.copy()
                simulated_env.score = current_score
                board_after_move, new_score, is_done, _ = simulated_env.step(move, generate=False)
                if not np.array_equal(board_state, board_after_move):
                    self.available_actions[move] = (board_after_move, new_score)

        self.pending_actions = list(self.available_actions.keys())

    def is_completely_expanded(self):
        if not self.available_actions:
            return False
        return all(action in self.children_nodes for action in self.available_actions)

    def is_leaf_node(self):
        return not self.is_completely_expanded()

class DecisionNode:
    def __init__(self, board_state, current_score, parent_node, action_taken):
        self.board_state = board_state
        self.current_score = current_score
        self.parent_node = parent_node
        self.action_taken = action_taken
        self.children_nodes = {}  # (position, value) -> DecisionNode
        self.visitation_count = 0
        self.accumulated_reward = 0.0
        self.is_expanded = False

    def is_leaf_node(self):
        return not self.is_expanded

    def is_fully_expanded(self, empty_positions):
        return len(self.children_nodes) == len(empty_positions) * 2  # Considering 2 and 4

# Main algorithm for the search
class TemporalMCTS:
    def __init__(self, game_env, value_predictor, num_iterations=50, exploration_factor=0.0, rollout_depth=10, discount_factor=1):
        self.game_env = game_env
        self.num_iterations = num_iterations
        self.exploration_factor = exploration_factor
        self.rollout_depth = rollout_depth
        self.discount_factor = discount_factor
        
        self.value_predictor = value_predictor
        self.min_score_encountered = float('inf')
        self.max_score_encountered = float('-inf')

    def generate_env_from_state(self, board_state, current_score):
        """
        Create a deep copy of the environment with a given board state and score.
        """
        simulated_env = copy.deepcopy(self.game_env)
        simulated_env.board = board_state.copy()
        simulated_env.score = current_score
        return simulated_env

    def evaluate_optimal_after_move_value(self, simulated_env, value_predictor):
        simulated_root_node = BaseNode(simulated_env.board.copy(), simulated_env.score, environment=simulated_env)
        if not simulated_root_node.available_actions:
            return 0
        
        best_value = float('-inf')
        for move, (board_after_move, new_score) in simulated_root_node.available_actions.items():
            move_reward = new_score - simulated_env.score
            estimated_value = move_reward + value_predictor.value(board_after_move)
            best_value = max(best_value, estimated_value)
        return best_value

    def select_best_child(self, node):
        # Select the best child using UCB (Upper Confidence Bound) formula
        best_ucb_value = -float("inf")
        best_child_node = None
        best_move = None
        for move, child_node in node.children_nodes.items():
            if child_node.visitation_count == 0:
                ucb_value = self.value_predictor.value(child_node.board_state)
            else:
                average_reward = child_node.accumulated_reward / child_node.visitation_count
                exploration_bonus = self.exploration_factor * math.sqrt(math.log(node.visitation_count) / child_node.visitation_count)
                ucb_value = average_reward + exploration_bonus
            if ucb_value > best_ucb_value:
                best_child_node = child_node
                best_move = move
                best_ucb_value = ucb_value
        return best_move, best_child_node

    def traverse(self, root_node):
        current_node = root_node
        simulated_env = self.generate_env_from_state(current_node.board_state, current_node.current_score)
        total_reward = 0
        while not current_node.is_leaf_node():

            if isinstance(current_node, BaseNode):
                best_move, _ = self.select_best_child(current_node)
                previous_score = simulated_env.score
                _, new_score, done, _ = simulated_env.step(best_move, generate=False)
                move_reward = new_score - previous_score
                total_reward += move_reward

                if best_move not in current_node.children_nodes:
                    current_node.children_nodes[best_move] = DecisionNode(simulated_env.board.copy(), new_score, parent_node=current_node, action_taken=best_move)
                current_node = current_node.children_nodes[best_move]

            elif isinstance(current_node, DecisionNode):
                possible_keys = list(current_node.children_nodes.keys())  # key: (position, value)
                action_weights = [0.9 if value == 2 else 0.1 for (_, value) in possible_keys]
                selected_key = random.choices(possible_keys, weights=action_weights, k=1)[0]

                current_node = current_node.children_nodes[selected_key]
                simulated_env = self.generate_env_from_state(current_node.board_state, current_node.current_score)

        return current_node, simulated_env, total_reward

    def expand_node(self, node, simulated_env):
        if simulated_env.is_game_over():
            return node, simulated_env

        if isinstance(node, BaseNode) and not node.children_nodes:
            for move, (board_after_move, new_score) in node.available_actions.items():
                random_decision_node = DecisionNode(board_after_move.copy(), new_score, parent_node=node, action_taken=move)
                node.children_nodes[move] = random_decision_node

        elif isinstance(node, DecisionNode) and not node.is_expanded:
            self.expand_random_decision_node(node)

    def simulate_rollout(self, node, simulated_env, total_reward):
        """
        Estimate the value of a node using the value_predictor
        """
        if isinstance(node, BaseNode):
            estimated_value = self.evaluate_optimal_after_move_value(simulated_env, self.value_predictor)
        elif isinstance(node, DecisionNode):
            estimated_value = self.value_predictor.value(node.board_state)
        else:
            estimated_value = 0

        estimated_value = total_reward + estimated_value
        # Normalize values
        if self.exploration_factor != 0:
            self.min_score_encountered = min(self.min_score_encountered, estimated_value)
            self.max_score_encountered = max(self.max_score_encountered, estimated_value)
            if self.max_score_encountered == self.min_score_encountered:
                normalized_value = 0.0
            else:
                normalized_value = 2 * (estimated_value - self.min_score_encountered) / (self.max_score_encountered - self.min_score_encountered) - 1
        else:
            normalized_value = estimated_value

        return normalized_value

    def propagate_backwards(self, node, reward):
        # Update statistics through the tree from the node upwards
        while node is not None:
            node.visitation_count += 1
            node.accumulated_reward += reward
            node = node.parent_node

    def expand_random_decision_node(self, node):
        available_positions = list(zip(*np.where(node.board_state == 0)))

        for position in available_positions:
            for value in [2, 4]:
                new_board_state = node.board_state.copy()
                new_board_state[position] = value
                key = (position, value)
                if key not in node.children_nodes:
                    random_node = BaseNode(new_board_state, node.current_score, parent_node=node, action_taken=key, environment=self.game_env)
                    node.children_nodes[key] = random_node

        node.is_expanded = True

    def execute_simulation(self, root_node):
        # Selection
        selected_node, simulated_env, total_reward = self.traverse(root_node)

        # Expansion
        self.expand_node(selected_node, simulated_env)

        # Rollout
        reward = self.simulate_rollout(selected_node, simulated_env, total_reward)

        # Backpropagation
        self.propagate_backwards(selected_node, reward)

    def best_action_distribution(self, root_node):
        """
        Computes the visit count distribution for each action at the root node.
        """
        total_visits = sum(child.visitation_count for child in root_node.children_nodes.values())
        visit_distribution = np.zeros(4)
        highest_visits = -1
        best_move = None
        for move, child_node in root_node.children_nodes.items():
            visit_distribution[move] = child_node.visitation_count / total_visits if total_visits > 0 else 0
            if child_node.visitation_count > highest_visits:
                highest_visits = child_node.visitation_count
                best_move = move
        return best_move, visit_distribution


def play_game():
    env = Game2048Env()
    state = env.reset()
    done = False
    score = 0  
    while not done:
        action = get_action(state, score)
        state, score, done, _ = env.step(action)
        print("Action:", action)
        print("State:\n", state)

    print("Game Over! Final Score:", score)


if __name__ == "__main__":
    play_game()