# Chain reaction

This was made to compete here https://www.hackerearth.com/problem/multiplayer/chainreaction/
The _chainreaction.py_ file implements the game. Given a state in stdin of type 

```
00 00 00 21 00  
00 00 00 00 00  
00 00 13 13 21  
00 00 00 00 00  
00 00 00 00 00  
1
```

it outputs a move of the type `0 0`. It performs alpha-beta pruning on a min-max algorithm. 

# Tournament

_tournament.py_ implements some sort of genetic algorithm to tune the parameters. It spawns some bots with different
configurations and perform mutation and crossover over the most fit. 