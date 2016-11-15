import logging
import sys
import time

RATIO_OF_DISCARDED_MOVESD = 4

CRISIS_GAME_RATIO = 4

MGE = 6200

MAX_CASCADE_DEPTH = 4
MAX_CASCADE_DEPTH_CRISIS = 2

FLOAT_INF = float("inf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """

    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__

@memoize
def side(c_i, r_i):
    return (r_i == 0) or (r_i == Game.SIZE - 1) or (c_i == 0) or (c_i == Game.SIZE - 1)

@memoize
def corner(c_i, r_i):
    return (r_i == 0 and c_i == 0) or (r_i == Game.SIZE - 1 and c_i == 0) or (
        r_i == 0 and c_i == Game.SIZE - 1) or (r_i == Game.SIZE - 1 and c_i == Game.SIZE - 1)

@memoize
def surroundings(position):
    cells = [(position[0] + 1, position[1]),
             (position[0], position[1] + 1),
             (position[0] - 1, position[1]),
             (position[0], position[1] - 1)]
    return [c for c in cells if ((0 <= c[0] < Game.SIZE) and (0 <= c[1] < Game.SIZE))]


def pretty_state(cells):
    s = ""
    for row in cells:
        for c in row:
            s += str(c[0]) + str(c[1]) + " "
        s += "\n"
    return s


class Game():
    SIZE = 5

    def __init__(self, state, max_cascade_depth=MAX_CASCADE_DEPTH):
        self.cells = state
        self.can_win = self.check_can_win()
        self.winner = None
        self.max_cascade_depth = max_cascade_depth

    def get_colour(self, position):
        return self.cells[position[0]][position[1]][0]

    def set_colour(self, position, colour):
        old_cell = self.cells[position[0]][position[1]]
        self.cells[position[0]][position[1]] = (colour, old_cell[1])
        return old_cell[0] != colour

    def get_quantity(self, position):
        return self.cells[position[0]][position[1]][1]

    def set_quantity(self, position, quantity):
        old_cell = self.cells[position[0]][position[1]]
        self.cells[position[0]][position[1]] = (old_cell[0], quantity)

    def increase_quantity(self, position):
        self.set_quantity(position, self.get_quantity(position) + 1)

    def move_allowed(self, position, colour):
        current_colour = self.get_colour(position)
        return current_colour == 0 or current_colour == colour

    def move(self, position, colour):
        # We get a copy of the game
        new_game = Game(self.copy_state(self.cells), max_cascade_depth=self.max_cascade_depth)
        new_game.__move__(position, colour)
        return new_game

    def __move__(self, position, colour, depth=0):
        if self.move_allowed(position, colour):
            self.increase_quantity(position)
            self.set_colour(position, colour)
            if depth < self.max_cascade_depth:
                self.cascade(position, depth)
        else:
            raise AssertionError(
                "Move not allowed " + str(colour) + " to " + str(position) + " in game " + str(self.cells))

    def cascade(self, lastmove, depth):
        colour_exploding = self.explodes(lastmove)
        if colour_exploding:
            self.reset(lastmove)
            for cell in surroundings(lastmove):
                converted = self.convert(cell, colour_exploding)
                self.__move__(cell, colour_exploding, depth + 1)
                if converted and self.check_ended_after_move_from_colour(colour_exploding):
                    self.winner = colour_exploding
                    return

    def reset(self, position):
        self.cells[position[0]][position[1]] = (0, 0)

    def convert(self, position, colour_exploding):
        return self.set_colour(position, colour_exploding)

    def explodes(self, position):
        n = self.get_quantity(position)
        return self.get_colour(position) if n >= len(surroundings(position)) else None

    def count(self):
        count_balls = {1: 0, 2: 0}
        #count_cells = {1: 0, 2: 0}
        count_corners_balls = {1: 0, 2: 0}
        count_edges_balls = {1: 0, 2: 0}
        for c, (r_i, c_i) in self.iter_cells():
            colour = c[0]
            if colour == 0:
                continue
            value = c[1]
            #count_cells[colour] += 1
            if corner(c_i, r_i):
                count_corners_balls[colour] += value
            elif side(c_i, r_i):
                count_edges_balls[colour] += value
            else:
                count_balls[colour] += value
        return count_balls, count_edges_balls, count_corners_balls

    def moves_for(self, colour):
        moves = []
        for c, pos in self.iter_cells():
            if c[0] == colour or c[0] == 0:
                moves.append(pos)
        return moves

    # @profile
    def check_ended_after_move_from_colour(self, colour):
        if not self.can_win:
            return False
        for c, _ in self.iter_cells():
            if c[0] != 0 and c[0] != colour:
                return False
        return True

    @staticmethod
    def state_from_string(state):
        rows = state.strip().split("\n")
        return [[(int(c[0]), int(c[1])) for c in r.split(" ")] for r in rows]

    @staticmethod
    def position_from_string(position):
        return tuple([int(p) for p in position.strip().split(" ")])

    def iter_cells(self):
        for row_index, row in enumerate(self.cells):
            for col_index, c in enumerate(row):
                yield c, (row_index, col_index)

    @staticmethod
    def copy_state(cells):
        newstate = []
        for row in cells:
            newrow = []
            for c in row:
                newrow.append(c[:])
            newstate.append(newrow)
        return newstate

    def check_can_win(self):
        seen = False
        for c, _ in self.iter_cells():
            if c[0] != 0:
                if seen:
                    return True
                else:
                    seen = True
        return False

class Player():
    def __init__(self, colour, max_depth=None, max_games_explored=MGE):
        self.max_games_explored = max_games_explored
        self.colour = colour
        self.max_depth = max_depth
        self.complexity = None
        self.games_explored = 0
        self.games_pruned = 0
        self.winning_games = 0
        self.crisis_games = 0

    def stats(self):
        return "ge:{}, gp:{}, wg:{}, md:{}, c:{}, cg:{}".format(self.games_explored, self.games_pruned,
                                                                self.winning_games, self.max_depth, self.complexity,
                                                                self.crisis_games)

    def pick_move(self, state):
        state = Game(state=state)
        self.complexity = len(state.moves_for(self.colour))
        if not self.max_depth:
            x = float(self.complexity)
            self.max_depth = \
                17.2117 - 0.898064 * x + 0.016048 * x * x
        return self.minmax(state, 0)

    def minmax(self, game, depth, alpha=-FLOAT_INF, beta=FLOAT_INF, is_max=True):
        self.games_explored += 1
        if self.games_explored > self.max_games_explored / CRISIS_GAME_RATIO:
            game.max_cascade_depth = MAX_CASCADE_DEPTH_CRISIS
            self.crisis_games += 1
        colour_here = self.colour_from_max(is_max)
        moves = game.moves_for(colour_here)
        limit = -FLOAT_INF if is_max else FLOAT_INF
        best_score_and_move = limit, None
        # We want to order the moves we have by how well perform with the heuristic
        scored_moves = []

        for move in moves:
            new_game_scored = game.move(move, colour_here)
            heuristic_score = self.heuristic(new_game_scored)
            scored_moves.append((heuristic_score, new_game_scored, move))
        depth_ = depth + 1
        sorted_moves = sorted(scored_moves, key=lambda x: x[0], reverse=is_max)
        num_moves_available = len(sorted_moves)

        for scored_move in sorted_moves[0:num_moves_available - num_moves_available/ RATIO_OF_DISCARDED_MOVESD]:
            new_game_score, new_game, move = scored_move
            if new_game.check_ended_after_move_from_colour(self.colour_from_max(is_max)):
                # The new game is ended, maximise the score for max/min and stop searching here
                best_score_and_move = -limit, move
                self.winning_games += 1
                break
            elif (depth_ >= self.max_depth) or self.games_explored > self.max_games_explored:
                # We reached a limit, just using heuristicd here under new_game_score
                pass
            else:
                # Go down using the current alpha and beta
                new_game_score, _ = self.minmax(new_game, depth_, alpha=alpha, beta=beta, is_max=not is_max)
            if (not best_score_and_move[1]) or \
                    (new_game_score > best_score_and_move[0] and is_max) or \
                    (new_game_score < best_score_and_move[0] and not is_max):
                best_score_and_move = new_game_score, move
                if is_max:
                    alpha = max(alpha, best_score_and_move[0])
                else:
                    beta = min(beta, best_score_and_move[0])
                if beta < alpha:
                    # print beta, alpha, colour_here, depth
                    self.games_pruned += 1
                    break
        assert best_score_and_move[1], "Not returning a move for game {} and depth {} and colour {}, instead {}".format(
            game.cells, depth, colour_here, best_score_and_move)
        # self.seen[entry_seen] = best_score_and_move
        logger.debug(
            "\n{} state:\n{}Best move {} with score {}".format(depth, pretty_state(game.cells), best_score_and_move[1],
                                                               best_score_and_move[0]))
        return best_score_and_move

    def colour_from_max(self, is_max):
        return self.colour if is_max else self.opponent_colour()

    def heuristic(self, game):
        count_balls, count_edges_balls, count_corners_balls = game.count()
        diff = 3 * (count_balls[self.colour] - count_balls[self.opponent_colour()]) \
               + 4.5 * (count_edges_balls[self.colour] - count_edges_balls[self.opponent_colour()]) \
               + 5.5 * (count_corners_balls[self.colour] - count_corners_balls[self.opponent_colour()])
        return diff

    def opponent_colour(self):
        return 3 - self.colour


if __name__ == "__main__":
    state_lines = 0
    state = ""
    colour = None
    while state_lines < 5:
        state += sys.stdin.readline()
        state_lines += 1
    colour = int(sys.stdin.readline())
    player = Player(colour)
    state = Game.state_from_string(state)
    game = Game(state)
    start_time = time.time()
    # cProfile.run('player.pick_move(state)', sort=1)
    score, move = player.pick_move(state)
    print(str(move[0]) + " " + str(move[1]))
    print(player.stats())
    print(score)
    print(time.time() - start_time)