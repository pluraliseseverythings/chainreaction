import copy
import random

import itertools
from multiprocessing import Pool, Value, Lock

from chainreaction import *

NEW_CHILDREN = 20

HS_TOURNAMENT = 0.1

INITIAL_MUTATION_STEPS = 20

SIGMA_FLOAT_MUTATION = 0.25
SIGMA_INT_MUTATION_RATIO = 3

MUTATION_PROBABILITY = 0.2

INITIAL_POPULATION_SIZE = 8
POPULATION_TARGET=10

EMPTY_GAME = [[(0, 0)]*5]*5

idd = 0

pool = None

def get_id():
    global idd
    idd += 1
    return idd

def play_game(p1, p2):
    game = Game(EMPTY_GAME, max_cascade_depth=sys.maxint)
    next_p = p1
    moves = 0
    while True:
        _, move = next_p.pick_move(game)
        #print "{} moving {}".format(next_p.colour, move)
        game = game.move(move, next_p.colour)
        #print pretty_state(game.cells)
        moves += 1
        if game.check_ended_after_move_from_colour(next_p.colour):
            #print "{} wins".format(next_p.colour)
            return next_p, p2 if next_p == p1 else p1, 100 - moves
        next_p = p2 if next_p == p1 else p1


def mutate_float(param):
    if random.random() < MUTATION_PROBABILITY:
        return random.normalvariate(param, SIGMA_FLOAT_MUTATION)
    return param


def mutate_int(num):
    if random.random() < MUTATION_PROBABILITY:
        return int(round(random.normalvariate(num, float(num) / SIGMA_INT_MUTATION_RATIO)))
    return num


def mutate(par):
    mutated = par
    mutated.h_final = (mutate_float(par.h_final[0]), mutate_float(par.h_final[1]), mutate_float(par.h_final[2]))
    mutated.h_initial = (mutate_float(par.h_initial[0]), mutate_float(par.h_initial[1]), mutate_float(par.h_initial[2]))
    mutated.empties_threshold = (mutate_int(par.empties_threshold))
    mutated.hurried_penalty = (mutate_float(par.hurried_penalty))
    mutated.depth_params = (mutate_float(par.depth_params[0]), mutate_float(par.depth_params[1]), mutate_float(par.depth_params[2]))
    return mutated


def initial_population():
    population = []
    for i in range(INITIAL_POPULATION_SIZE - 1):
        par = Params()
        for i in range(INITIAL_MUTATION_STEPS):
            par = mutate(par)
        par.id = get_id()
        population.append(par)
    population.append(Params())
    return population


def tournament(population):
    global pool
    if not pool:
        pool = Pool()
    scores = {}
    for p in population:
        scores[p] = 0
    results = pool.map(play_and_add, itertools.combinations(population, 2))
    # for par1, par2 in itertools.combinations(population, 2):
    #       play_and_add((par1, par2, scores))
    for w, l, s in results:
        scores[w.p] += s
        scores[l.p] -= s
    return scores


def play_and_add(p):
    par1, par2 = p
    winner, loser, score = play_game(Player(1, parameters=par1, hs=HS_TOURNAMENT),
                                     Player(2, parameters=par2, hs=HS_TOURNAMENT))
    print "{} wins, {} loses. Score: {}".format(winner.p.id, loser.p.id, score)
    return (winner, loser, score)


def select_survivors(scored_population):
    sorted1 = sorted(scored_population.items(), key=lambda x: -x[1])
    return sorted1[0:int(round(len(scored_population) * 0.6 + 1))]


def combine(parent1, parent2):
    child = copy.deepcopy(parent1)
    if random.random < 0.5:
        child.h_final = parent2.h_final
    if random.random < 0.5:
        child.h_initial = parent2.h_initial
    if random.random < 0.5:
        child.empties_threshold = parent2.empties_threshold
    if random.random < 0.5:
        child.depth_params = parent2.depth_params
    if random.random < 0.5:
        child.hurried_penalty = parent2.hurried_penalty
    child.id = get_id()
    return child


def crossover(survivors, needed):
    list_of_prob = []
    children = []
    for e, score in survivors:
        list_of_prob.extend([e]*int(round(score/10)))

    while len(children) < needed:
        parent1 = random.choice(list_of_prob)
        parent2 = random.choice(list_of_prob)
        child = mutate(combine(parent1, parent2))
        children.append(child)
    return children


def generate_fittest():
    population = initial_population()
    while True:
        print "Pop size: {}".format(len(population))
        scored_population = tournament(population)
        survivors = select_survivors(scored_population)
        print "Survivors: "
        for k, s in survivors:
            print "Score {} for {}".format(s, str(k))
        children = crossover(survivors, POPULATION_TARGET - len(survivors))
        population = [s[0] for s in survivors] + children
        print "Population after crossover: "
        for k in population:
            print str(k)


if __name__ == "__main__":
    generate_fittest()
