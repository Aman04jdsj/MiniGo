import math
from copy import deepcopy

from numpy.random import normal

DEBUG = False


def read_input(n, path="input.txt"):
    """
    Read the input.txt file and generate previous and current board configuration

    :param n: Size of go game
    :param path: Path of the input file
    :return (piece_type, previous_board, board): The piece to play, previous and current board configs
    """
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]

        return piece_type, previous_board, board


def write_output(result, path="output.txt"):
    """
    Write the next move to the output.txt file

    :param result: The tuple(i, j) of next move or PASS
    :param path: Path of the output file
    """
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)
    return


def read_count(path="count.txt"):
    count = 1
    try:
        with open(path, "r") as f:
            count += int(f.readline().strip('\n'))
    except FileNotFoundError:
        return count
    return count


def write_count(count, path="count.txt"):
    with open(path, "w") as f:
        f.write(str(count))
    return


def write_game(piece_type, prev_board, board, action):
    if piece_type == 2:
        with open("white_games.txt", "a") as f:
            f.writelines(prev_board+'\n')
            f.write(str(action)+'\n')
            f.writelines(board+'\n')
        return
    with open("black_games.txt", "a") as f:
        f.writelines(prev_board+'\n')
        f.write(str(action)+'\n')
        f.writelines(board+'\n')
    return


def draw_board(board):
    """
    Visualize the board.

    :return: None
    """
    res = '-' * len(board) * 2 + '\n'
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                res += '  '
            elif board[i][j] == 1:
                res += 'X '
            else:
                res += 'O '
        res += '\n'
    res += '-' * len(board) * 2 + '\n'
    return res


class Go:
    def __init__(self, n):
        """
        Go game class.
        """
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = 2
        self.size = n
        self.dead_pieces = []
        self.black_killed = 0
        self.white_killed = 0
        self.black_pieces = dict()
        self.white_pieces = dict()
        self.black_pieces_cnt = 0
        self.white_pieces_cnt = 0
        self.empty_cnt = 0
        self.move_cnt = 0
        self.max_moves = n * n - 1
        self.komi = n / 2
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        self.previous_board = deepcopy(board)
        self.valid_actions = dict({str((i, j)): 1 for i in range(n) for j in range(n)})

    def is_board_equal(self, board1, board2):
        """
        Check if given two board configs are equal

        :param board1: First board
        :param board2: Second board
        :return: True if the 2 configs are equal else False
        """
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_game(self):
        """
        Returns a deepcopy of the game config
        """
        return deepcopy(self)

    def find_neighbours(self, i, j):
        """
        Find all the neighbours of a given piece

        :param i: Row number of the piece
        :param j: Column number of the piece
        :return neighbours: List of tuples of all the neighbours of (i, j)
        """
        neighbours = []
        if i > 0:
            neighbours.append((i - 1, j))
        if i < self.size - 1:
            neighbours.append((i + 1, j))
        if j > 0:
            neighbours.append((i, j - 1))
        if j < self.size - 1:
            neighbours.append((i, j + 1))
        return neighbours

    def find_neighbour_allies(self, i, j):
        """
        Return the allies of a given piece amongst its neighbours

        :param i: Row number of the piece
        :param j: Column number of the piece
        :return neighbour_allies: List of tuples of all neighbouring allies of (i, j)
        """
        board = self.board
        neighbours = self.find_neighbours(i, j)
        neighbour_allies = []
        for neighbour in neighbours:
            if board[i][j] == board[neighbour[0]][neighbour[1]]:
                neighbour_allies.append(neighbour)
        return neighbour_allies

    def find_connected_allies(self, i, j):
        """
        Return the allies of a given piece within its connected component

        :param i: Row number of the piece
        :param j: Column number of the piece
        :return connected_allies: List of tuples of all allies of (i, j) in its connected component
        """
        visited = dict()
        queue = [(i, j)]
        connected_allies = []
        while queue:
            ally = queue.pop(0)
            connected_allies.append(ally)
            visited[str(ally)] = 1
            neighbour_allies = self.find_neighbour_allies(ally[0], ally[1])
            for n_ally in neighbour_allies:
                if str(n_ally) not in visited:
                    visited[str(n_ally)] = 1
                    queue.append(n_ally)
        return connected_allies

    def has_liberty(self, i, j):
        """
        Check if the given position has liberty

        :param i: Row number of the piece
        :param j: Column number of the piece
        :return: True if any of the connected allies of (i, j) has a neighbouring empty space, else False
        """
        connected_allies = self.find_connected_allies(i, j)
        board = self.board
        for ally in connected_allies:
            neighbours = self.find_neighbours(ally[0], ally[1])
            for neighbour in neighbours:
                if board[neighbour[0]][neighbour[1]] == self.EMPTY:
                    return True
        return False

    def find_all_liberties(self):
        """
        Return a mapping of all group sizes to corresponding liberties for a given piece type

        :return liberties: Mapping of all group sizes to corresponding liberties
        """
        visited = dict()
        black_liberties = dict()
        white_liberties = dict()
        board = self.board
        for location in self.black_pieces.keys():
            if location not in visited:
                i, j = map(int, location.strip(")(").split(","))
                connected_allies = self.find_connected_allies(i, j)
                group_size = len(connected_allies)
                liberty = 0
                visited_neighbours = dict()
                for ally in connected_allies:
                    neighbours = self.find_neighbours(ally[0], ally[1])
                    for neighbour in neighbours:
                        if str(neighbour) not in visited_neighbours:
                            if board[neighbour[0]][neighbour[1]] == self.EMPTY:
                                liberty += 1
                            visited_neighbours[str(neighbour)] = 1
                    visited[str(ally)] = 1
                black_liberties[location] = {group_size: liberty}
        for location in self.white_pieces.keys():
            if location not in visited:
                i, j = map(int, location.strip(")(").split(","))
                connected_allies = self.find_connected_allies(i, j)
                group_size = len(connected_allies)
                liberty = 0
                visited_neighbours = dict()
                for ally in connected_allies:
                    neighbours = self.find_neighbours(ally[0], ally[1])
                    for neighbour in neighbours:
                        if str(neighbour) not in visited_neighbours:
                            if board[neighbour[0]][neighbour[1]] == self.EMPTY:
                                liberty += 1
                            visited_neighbours[str(neighbour)] = 1
                    visited[str(ally)] = 1
                white_liberties[location] = {group_size: liberty}
        return black_liberties, white_liberties

    def find_dead_pieces(self, piece_type):
        """
        Return the dead pieces of a given piece type

        :param piece_type: 1('X') or 2('O')
        :return dead_pieces: List of tuples of all dead pieces of type piece_type
        """
        dead_pieces = []
        visited = dict()
        board = self.board
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    if str((i, j)) not in visited:
                        if not self.has_liberty(i, j):
                            dead_connected_allies = self.find_connected_allies(i, j)
                            for ally in dead_connected_allies:
                                visited[str(ally)] = 1
                            dead_pieces += dead_connected_allies
        return dead_pieces

    def remove_dead_pieces(self, piece_type):
        """
        Removes the dead pieces of a given piece type from the board

        :param piece_type: 1('X') or 2('O')
        """
        dead_pieces = self.find_dead_pieces(piece_type)
        board = self.board
        for piece in dead_pieces:
            board[piece[0]][piece[1]] = self.EMPTY
            if piece_type == self.BLACK:
                del self.black_pieces[str(piece)]
                self.black_killed += 1
            else:
                del self.white_pieces[str(piece)]
                self.white_killed += 1
            self.valid_actions[str(piece)] = 1
            self.empty_cnt += 1
        self.update_board(board)
        return

    def update_board(self, board):
        """
        Update the board configuration with the given board

        :param board: The new board config
        """
        self.board = board
        return

    def place_piece(self, i, j, piece_type):
        """
        Place a piece of given piece type at given location if valid, and update the board

        :param i: Row number of new piece
        :param j: Column number of new piece
        :param piece_type: 1('X') or 2('O')
        :return: True if the placement is valid, else False
        """
        board = self.board

        valid_place = self.is_valid(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        self.remove_dead_pieces(3-piece_type)
        del self.valid_actions[str((i, j))]
        if piece_type == self.WHITE:
            self.white_pieces[str((i, j))] = 1
            self.white_pieces_cnt += 1
        else:
            self.black_pieces[str((i, j))] = 1
            self.black_pieces_cnt += 1
        self.empty_cnt -= 1
        return True

    def is_valid(self, i, j, piece_type):
        """
        Check if a piece of given piece type can be placed at given location

        :param i: Row number of new piece
        :param j: Column number of new piece
        :param piece_type: 1('X') or 2('O')
        :return: True if the placement is valid, else False
        """
        board = self.board
        n = self.size
        if not (0 <= i < n):
            return False
        if not (0 <= j < n):
            return False
        if board[i][j] != self.EMPTY:
            return False
        game_cpy = self.copy_game()
        board_cpy = game_cpy.board
        board_cpy[i][j] = piece_type
        game_cpy.update_board(board_cpy)
        if game_cpy.has_liberty(i, j):
            return True
        else:
            game_cpy.remove_dead_pieces(3 - piece_type)
            if not game_cpy.has_liberty(i, j):
                return False
            else:
                if self.dead_pieces and self.is_board_equal(self.previous_board, game_cpy.board):
                    return False
        return True

    def did_game_end(self, action="MOVE"):
        """
        Check if the game has ended

        :param action: The action to get to the current state
        :return: True if the game has ended, else False
        """
        if self.move_cnt >= self.max_moves:
            return True
        if self.is_board_equal(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def set_board(self, piece_type, previous_board, board):
        """
        Set the board based on the given current and previous board configurations

        :param piece_type: 1('X') or 2('O')
        :param previous_board: Configuration of board before the previous move
        :param board: Board configuration after the previous move
        :return opponent_move: "MOVE" if board is the initial state, else previous move of opponent
        """
        opponent_move = "PASS"
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.dead_pieces.append((i, j))
                    self.valid_actions[str((i, j))] = 1
                if board[i][j] == 3-piece_type and previous_board[i][j] != 3-piece_type:
                    opponent_move = (i, j)
                if board[i][j] != self.EMPTY:
                    del self.valid_actions[str((i, j))]
                    if board[i][j] == self.BLACK:
                        self.black_pieces[str((i, j))] = 1
                        self.black_pieces_cnt += 1
                    elif board[i][j] == self.WHITE:
                        self.white_pieces[str((i, j))] = 1
                        self.white_pieces_cnt += 1
                else:
                    self.empty_cnt += 1
        if self.white_pieces_cnt == 0 and self.black_pieces_cnt <= 1:
            if self.black_pieces_cnt == 0:
                opponent_move = "MOVE"
                write_count(0)
            else:
                self.move_cnt = 1
                write_count(1)
        else:
            self.move_cnt = read_count()
        self.previous_board = previous_board
        self.board = board
        return opponent_move


class AlphaBeta:
    def __init__(self, piece_type):
        self.piece_type = piece_type
        self.BLACK = 1
        self.WHITE = 2

    def max_agent(self, go, depth, alpha, beta, action):
        if depth == 0 or go.did_game_end(action):
            return "PASS", self.heuristic(go)
        max_score = -math.inf
        next_action = "PASS"
        for location in go.valid_actions.keys():
            i, j = map(int, location.strip(')(').split(', '))
            if go.is_valid(i, j, self.piece_type):
                go_copy = go.copy_game()
                _ = go_copy.place_piece(i, j, self.piece_type)
                go_copy.move_cnt += 1
                _, cur_score = self.min_agent(go_copy, depth - 1, alpha, beta, "MOVE")
                if DEBUG:
                    print("\t"*(4-depth), "Depth: ", depth, " Action checked: ", (i, j), " Heuristic val: ", cur_score, " Max agent")
                if cur_score > max_score:
                    max_score = cur_score
                    next_action = (i, j)
                if max_score > beta:
                    break
                alpha = max(alpha, max_score)
        if max_score == -math.inf:
            max_score = 0
        return next_action, max_score

    def min_agent(self, go, depth, alpha, beta, action):
        if depth == 0 or go.did_game_end(action):
            return "PASS", self.heuristic(go)
        min_score = math.inf
        next_action = "PASS"
        for location in go.valid_actions.keys():
            i, j = map(int, location.strip(')(').split(', '))
            if go.is_valid(i, j, 3 - self.piece_type):
                go_copy = go.copy_game()
                _ = go_copy.place_piece(i, j, 3 - self.piece_type)
                go_copy.move_cnt += 1
                _, cur_score = self.max_agent(go_copy, depth - 1, alpha, beta, "MOVE")
                if DEBUG:
                    print("\t" * (4 - depth), "Depth: ", depth, " Action checked: ", (i, j), " Heuristic val: ", cur_score, " Min agent")
                if cur_score < min_score:
                    min_score = cur_score
                    next_action = (i, j)
                if min_score < alpha:
                    break
                beta = min(beta, min_score)
        if min_score == math.inf:
            min_score = 0
        return next_action, min_score

    def heuristic(self, go):
        """
        Get the heuristic value of a given board w.r.t a piece type

        :param go: Go instance.
        :return: Heuristic value of board
        """
        black_liberty_map, white_liberty_map = go.find_all_liberties()
        black_endangered = 0
        white_endangered = 0
        black_liberties = 0
        white_liberties = 0
        black_cnt = 0
        white_cnt = 0
        for _, pair in black_liberty_map.items():
            for k, v in pair.items():
                if v == 1:
                    black_endangered += k
                else:
                    black_liberties += v
                black_cnt += 1
        for _, pair in white_liberty_map.items():
            for k, v in pair.items():
                if v == 1:
                    white_endangered += k
                else:
                    white_liberties += v
                white_cnt += 1
        if self.piece_type == self.WHITE:
            return (white_liberties - black_liberties)*normal(1, 0.1) + \
                   (black_endangered - white_endangered)*normal(1, 0.1) + \
                   white_cnt + go.komi - black_cnt + (go.black_killed - go.white_killed)
        return (black_liberties - white_liberties)*normal(1, 0.1) + \
            (white_endangered - black_endangered)*normal(1, 0.1) + \
            black_cnt - white_cnt - go.komi + (go.white_killed - go.black_killed)*1.27

    def get_action(self, go, prev_action):
        """
        Get the next action

        :param go: Go instance.
        :param prev_action: Action to get to current state
        :return: (row, column) coordinate of next move
        """
        empty = go.empty_cnt
        if empty > 20:
            alpha_beta_depth = 2
        elif 15 < empty <= 20:
            alpha_beta_depth = 3
        elif 10 < empty <= 15:
            alpha_beta_depth = 4
        else:
            alpha_beta_depth = 5
        action, _ = self.max_agent(go, alpha_beta_depth, -math.inf, math.inf, prev_action)
        go.move_cnt += 1
        return action


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = read_input(N)
    prev = deepcopy(board)
    go = Go(N)
    prev_action = go.set_board(piece_type, previous_board, board)
    player = AlphaBeta(piece_type)
    action = player.get_action(go, "MOVE" if prev_action != "PASS" else prev_action)
    if action != "PASS":
        go.place_piece(action[0], action[1], piece_type)
    write_game(piece_type, draw_board(prev), draw_board(go.board), action)
    write_output(action)
    write_count(go.move_cnt)

