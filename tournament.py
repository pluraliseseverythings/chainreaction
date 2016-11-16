from chainreaction import *

EMPTY_GAME = [[(0, 0)]*5]*5


def play_game():
    game = Game(EMPTY_GAME, max_cascade_depth=sys.maxint)
    next_p = p1
    while True:
        _, move = next_p.pick_move(game)
        print "{} moving {}".format(next_p.colour, move)
        game = game.move(move, next_p.colour)
        print pretty_state(game.cells)
        if game.check_ended_after_move_from_colour(next_p.colour):
            print "{} wins".format(next_p.colour)
            break
        next_p = p2 if next_p == p1 else p1


if __name__ == "__main__":
    p1 = Player(1)
    p2 = Player(2)
    play_game()
