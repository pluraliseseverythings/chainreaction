import unittest

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

    def test_state_from_string(self):
        self.assertEqual(Game.state_from_string(GameTest.INPUT),
                         [[[0, 0], [0, 0], [0, 0], [2, 1], [0, 0]],
                          [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                          [[0, 0], [0, 0], [1, 3], [1, 3], [2, 1]],
                          [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                          [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])

    def test_game_won(self):
        game = Game(state=Game.state_from_string(GameTest.WON))

        self.assertTrue(game.check_ended())
        game = Game(state=Game.state_from_string(GameTest.INPUT))
        self.assertFalse(game.check_ended())
        game = Game(state=Game.state_from_string(EMPTY_INPUT))
        self.assertFalse(game.check_ended())

    def test_position_from_string(self):
        self.assertEqual(Game.position_from_string(GameTest.MOVE), [2, 2])

    def test_move(self):
        game = Game(state=Game.state_from_string(GameTest.INPUT))
        move = Game.position_from_string(GameTest.MOVE)
        new_game = game.move(move, GameTest.COLOUR)
        self.assertEquals(new_game.cells, Game.state_from_string(GameTest.EXPEC))

    def test_moves_for(self):
        game = Game(state=Game.state_from_string(GameTest.INPUT))
        self.assertEquals(game.moves_for(GameTest.COLOUR),
                          [[0, 0], [0, 1], [0, 2],         [0, 4],
                           [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                           [2, 0], [2, 1], [2, 2], [2, 3],
                           [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
                           [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]])
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
            game.move(m)

    def test_count(self):
        game = Game(state=Game.state_from_string(GameTest.INPUT))
        self.assertEquals(game.count(), {1: 6, 2: 2}, {1: 2, 2: 2}


class PlayerTest(unittest.TestCase):
    def test_pick_move(self):
        p = Player(GameTest.COLOUR)
        m = p.pick_move(Game.state_from_string(GameTest.INPUT))
        self.assertEqual(m, [3,0])

    def test_pick_move_empty(self):
        p = Player(GameTest.COLOUR)
        m = p.pick_move(Game.state_from_string(EMPTY_INPUT))
        self.assertEqual(m, [0,0])

    def test_pick_move_2(self):
        p = Player(1)
        m = p.pick_move(Game.state_from_string(STATE_2))
        self.assertEqual(m, [4,4])

