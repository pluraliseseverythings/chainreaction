import copy
import cProfile
import sys
import time

FLOAT_INF = float("inf")

def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__


def surroundings(position):
    return surroundings_tuple(tuple(position))

@memoize
def surroundings_tuple(position):
    cells = [[position[0] + 1, position[1]],
             [position[0], position[1] + 1],
             [position[0] - 1, position[1]],
             [position[0], position[1] - 1]]
    return [c for c in cells if ((0 <= c[0] < Game.SIZE) and (0 <= c[1] < Game.SIZE))]

class Game():
    SIZE = 5
    EMPTY = [[[0, 0]] * SIZE] * SIZE

    def __init__(self, state):
        self.cells = state
        self.winner = None

    def get_colour(self, position):
        return self.cells[position[0]][position[1]][0]

    def set_colour(self, position, colour):
        self.cells[position[0]][position[1]][0] = colour

    def get_quantity(self, position):
        return self.cells[position[0]][position[1]][1]

    def set_quantity(self, position, quantity):
        self.cells[position[0]][position[1]][1] = quantity

    def increase_quantity(self, position):
        self.set_quantity(position, self.get_quantity(position) + 1)

    def move_allowed(self, position, colour):
        current_colour = self.get_colour(position)
        return current_colour == 0 or current_colour == colour

    def move(self, position, colour):
        new_game = Game(self.copy_state(self.cells))
        new_game.__move__(position, colour)
        return new_game

    def __move__(self, position, colour):
        if self.move_allowed(position, colour):
            self.increase_quantity(position)
            self.set_colour(position, colour)
            self.cascade(position)
        else:
            raise AssertionError("Move not allowed " + str(colour) + " to " + str(position) + " in game " + str(self.cells))

    def cascade(self, lastmove):
        colour_exploding = self.explodes(lastmove)
        if colour_exploding:
            self.reset(lastmove)
            for cell in surroundings(lastmove):
                self.convert(cell, colour_exploding)
                self.__move__(cell, colour_exploding)
                if self.check_ended():
                    self.winner = colour_exploding
                    return

    def reset(self, position):
        self.cells[position[0]][position[1]] = [0, 0]

    def convert(self, position, colour_exploding):
        self.set_colour(position, colour_exploding)

    def explodes(self, position):
        n = self.get_quantity(position)
        return self.get_colour(position) if n >= len(surroundings(position)) else None

    def count(self):
        count_balls = {1:0, 2:0}
        count_cells = {1:0, 2:0}
        for c, _ in self.iter_cells():
            colour = c[0]
            if colour == 0:
                continue
            value = c[1]
            count_balls[colour] += value
            count_cells[colour] += 1
        return count_balls, count_cells

    def moves_for(self, colour):
        moves = []
        for c, pos in self.iter_cells():
            if c[0] == colour or c[0] == 0:
                moves.append(pos)
        return moves

    def check_ended(self):
        seen = [0, 0]
        for row in self.cells:
            for c in row:
                colour = c[0]
                q = c[1]
                if colour != 0:
                    seen[colour - 1] += q
                    if 1 < seen[colour - 1] == seen[0] + seen[1]:
                        return False
        return len(seen) == 1

    @staticmethod
    def surroundings(position):
        cells = [[position[0] + 1, position[1]],
                 [position[0], position[1] + 1],
                 [position[0] - 1, position[1]],
                 [position[0], position[1] - 1]]
        return [c for c in cells if ((0 <= c[0] < Game.SIZE) and (0 <= c[1] < Game.SIZE))]

    @staticmethod
    def state_from_string(state):
        rows = state.strip().split("\n")
        return [[[int(c[0]), int(c[1])] for c in r.split(" ")] for r in rows]

    @staticmethod
    def position_from_string(position):
        return [int(p) for p in position.strip().split(" ")]

    def iter_cells(self):
        row_index = 0
        col_index = 0
        for row in self.cells:
            for c in row:
                yield c, [row_index, col_index]
                col_index += 1
            col_index = 0
            row_index += 1

    @staticmethod
    def copy_state(cells):
        newstate = []
        for row in cells:
            newrow = []
            for c in row:
                newrow.append(c[:])
            newstate.append(newrow)
        return newstate


def to_tuple(cells):
    t_cells = []
    for row in cells:
        t_row = []
        for c in row:
            t_row.append(tuple(c))
        t_cells.append(tuple(t_row))
    return tuple(t_cells)



class Player():
    MAX_DEPTH = 4

    def __init__(self, colour, max_depth=MAX_DEPTH, max_games_explored=20000):
        self.max_games_explored = max_games_explored
        self.colour = colour
        self.games_explored = 0
        self.max_depth = max_depth
        self.num_seen = 0

    def pick_move(self, state):
        self.seen = {}
        state = Game(state=state)
        return self.minmax(state, 0)

    def minmax(self, game, depth, alpha=-FLOAT_INF, beta=FLOAT_INF, is_max=True):
        # entry_seen = (is_max, to_tuple(game.cells))
        # if entry_seen in self.seen:
        #     seen_ = self.seen[entry_seen]
        #     self.num_seen += 1
        #     #print "Move " + str(seen_) + " from seen state " + str(entry_seen)
        #     return seen_
        #print depth, alpha, beta
        self.games_explored += 1
        colour_here = self.colour if is_max else self.opponent_colour()
        moves = game.moves_for(colour_here)
        limit = -FLOAT_INF if is_max else FLOAT_INF
        best_score_and_move = limit, None
        scored_moves = []
        for move in moves:
            new_game_scored = game.move(move, colour_here)
            scored_moves.append((self.heuristic(new_game_scored), new_game_scored, move))

        for scored_move in sorted(scored_moves, key=lambda x: x[0], reverse=is_max):
            new_game_score, new_game, move = scored_move
            depth_ = depth + 1
            if new_game.check_ended():
                best_score_and_move = -limit, move
                break
            elif (depth_ >= self.max_depth) or self.games_explored > self.max_games_explored:
                best_score_and_move = new_game_score, move
            else:
                new_game_score, _ = self.minmax(new_game, depth_, alpha=alpha, beta=beta, is_max=not is_max)
            best_score = best_score_and_move[0]
            if not best_score_and_move[1] or (new_game_score > best_score and is_max) or (new_game_score < best_score):
                best_score_and_move = new_game_score, move
                if is_max:
                    alpha = max(alpha, best_score)
                else:
                    beta = min(beta, best_score)
                if beta < alpha:
                    #print beta, alpha, colour_here, depth
                    break
        assert best_score_and_move[1], "Not returning a move for game {} and depth {} and colour {}, instead {}".format(game.cells, depth, colour_here, best_score_and_move)
        #self.seen[entry_seen] = best_score_and_move
        return best_score_and_move

    def heuristic(self, game):
        count_balls, count_cells = game.count()
        # about_to_explode = 0
        # for c, pos in game.iter_cells():
        #     if len(surroundings(pos)) <= game.get_quantity(pos) + 1:
        #         about_to_explode += 1 if c == self.colour else - 1
        diff = count_balls[self.colour] - count_balls[self.opponent_colour()] - (count_cells[self.colour])/2
        return diff #+ about_to_explode/2

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
    player = Player(colour, 4, max_games_explored=9000)
    state = Game.state_from_string(state)
    game = Game(state)
    start_time = time.time()
    #cProfile.run('player.pick_move(state)', sort=1)
    score, move = player.pick_move(state)
    print(str(move[0]) + " " + str(move[1]))
    print(player.games_explored)
    print(player.num_seen)
    print(score)
    print(time.time() - start_time)


