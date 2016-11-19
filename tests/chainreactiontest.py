import unittest
import cProfile
from chainreaction import *

STATE_1 = [[[1, 1], [0, 0], [1, 1], [1, 2], [0, 0]],
           [[1, 1], [1, 2], [1, 2], [0, 0], [0, 0]],
           [[1, 2], [1, 1], [1, 3], [1, 3], [2, 1]],
           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]

STATE_2 = "11 22 00 21 00\n" \
          "22 21 21 23 11\n" \
          "11 13 12 13 11\n" \
          "11 21 13 11 12\n" \
          "11 12 12 00 11"

STATE_3 = "00 00 00 00 00\n" \
          "00 00 00 00 00\n" \
          "00 00 00 00 00\n" \
          "00 00 00 23 12\n" \
          "00 00 00 21 11"

STATE_6 = "00 21 12 12 11\n" \
          "21 23 12 13 00\n" \
          "22 00 21 12 12\n" \
          "21 23 12 00 21\n" \
          "21 22 22 21 21"

STATE_WUT = "00 12 11 12 11\n11 12 13 00 12\n11 13 13 11 12\n21 00 22 13 00\n21 21 22 22 11"

STATE_WIN = [("11 12 12 12 00\n12 11 13 12 12\n11 00 13 00 12\n11 13 13 12 22\n11 11 12 12 11", 2, (3, 4))]

STATES = [("11 11 22 00 21\n11 23 00 21 11\n11 00 21 00 12\n11 00 23 00 00\n11 00 00 11 21", 1, (1, 2), True),
          ("00 12 11 22 21\n11 13 13 11 22\n11 00 00 00 00\n21 00 00 00 22\n21 22 22 22 21", 1, (1, 1), True)]


EMPTY_INPUT = "00 00 00 00 00\n" \
              "00 00 00 00 00\n" \
              "00 00 00 00 00\n" \
              "00 00 00 00 00\n" \
              "00 00 00 00 00"

class GameTest(unittest.TestCase):
    INPUT = "00 00 00 21 00\n" \
            "00 00 00 00 00\n" \
            "00 00 13 13 21\n" \
            "00 00 00 00 00\n" \
            "00 00 00 00 00"

    EXPEC = "00 00 00 21 00\n" \
            "00 00 11 11 00\n" \
            "00 11 11 00 12\n" \
            "00 00 11 11 00\n" \
            "00 00 00 00 00"

    WON   = "00 00 00 11 00\n" \
            "00 00 11 11 00\n" \
            "00 11 11 00 12\n" \
            "00 00 11 11 00\n" \
            "00 00 00 00 00"

    MOVE = "2 2"

    COLOUR = 1

    OPPONENT = 2

    def test_game_from_string(self):
        self.assertEqual(Game.state_from_string(GameTest.INPUT),
                         [[(0, 0), (0, 0), (0, 0), (2, 1), (0, 0)],
                          [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
                          [(0, 0), (0, 0), (1, 3), (1, 3), (2, 1)],
                          [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
                          [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]])

    def test_game_won(self):
        game = Game(state=Game.state_from_string(GameTest.WON))
        self.assertTrue(game.check_ended_after_move_from_colour(1))
        game = Game(state=Game.state_from_string(GameTest.INPUT))
        self.assertFalse(game.check_ended_after_move_from_colour(1))
        game = Game(state=Game.state_from_string(EMPTY_INPUT))
        self.assertFalse(game.check_ended_after_move_from_colour(1))

    def test_position_from_string(self):
        self.assertEqual(Game.position_from_string(GameTest.MOVE), (2, 2))

    def test_move(self):
        game = Game(state=Game.state_from_string(GameTest.INPUT))
        move = Game.position_from_string(GameTest.MOVE)
        new_game = game.move(move, GameTest.COLOUR)
        self.assertEquals(new_game.cells, Game.state_from_string(GameTest.EXPEC))

    def test_move_2(self):
        game = Game(state=Game.state_from_string(STATE_WUT))
        move = (4, 3)
        new_game = game.move(move, 2)
        print pretty_state(new_game.cells)

    def test_moves_for(self):
        game = Game(state=Game.state_from_string(GameTest.INPUT))
        self.assertEquals(game.moves_for(GameTest.COLOUR),
                          [(0, 0), (0, 1), (0, 2),         (0, 4),
                           (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                           (2, 0), (2, 1), (2, 2), (2, 3),
                           (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                           (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)])
        self.assertEquals(len(game.moves_for(GameTest.OPPONENT)), 23)
        game = Game(state=Game.state_from_string(GameTest.EXPEC))
        self.assertEquals(len(game.moves_for(GameTest.COLOUR)), 24)
        self.assertEquals(len(game.moves_for(GameTest.OPPONENT)), 18)
        game = Game(state=Game.state_from_string(GameTest.WON))
        self.assertEquals(len(game.moves_for(GameTest.COLOUR)), 25)
        self.assertEquals(len(game.moves_for(GameTest.OPPONENT)), 17)

    def test_tricky_move_1(self):
        game = Game(STATE_1)
        self.assertFalse([0,2] in game.moves_for(2))

    def moves_are_ok(self):
        game = Game(STATE_1)
        moves = game.moves_for(2)
        for m in moves:
            game.move(m, 2)


class PlayerTest(unittest.TestCase):
    def test_pick_move(self):
        p = Player(GameTest.COLOUR)
        m = p.pick_move(Game.game_from_string(GameTest.INPUT))
        self.assertEqual(m[1], (2,2))

    def test_pick_move_empty(self):
        p = Player(GameTest.COLOUR)
        m = p.pick_move(Game.game_from_string(EMPTY_INPUT))
        self.assertEqual(m[1], (0,0))

    def test_pick_move_2(self):
        p = Player(1)
        m = p.pick_move(Game.game_from_string(STATE_2))
        self.assertEqual(m[1], (2,3))

    def test_pick_move_3(self):
        p = Player(1)
        m = p.pick_move(Game.game_from_string(STATE_3))
        self.assertEqual(m[0], (float("inf")))

    def test_pick_move_s0(self):
        cells, pl, move, win = STATES[0]
        self.r(cells, move, pl, win)

    def test_pick_move_s1(self):
        cells, pl, move, win = STATES[1]
        self.r(cells, move, pl, win)

    def r(self, cells, move, pl, win):
        p = Player(pl)
        s, m = p.pick_move(Game.game_from_string(cells))
        print m
        if win:
            self.assertEqual(m, move)
        else:
            self.assertNotEqual(m, move)

    def test_won(self):
        cells, pl, move = STATE_WIN[0]
        g = Game(Game.state_from_string(cells)).move(move, pl)
        self.assertTrue(g.check_ended_after_move_from_colour(pl))




