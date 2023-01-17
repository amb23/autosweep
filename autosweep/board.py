
from enum import Enum
from typing import Tuple, Set
import numpy as np
from dataclasses import dataclass, field



@dataclass
class Cell:
    revealed: bool
    is_mine: bool
    loc: int
    adjacent: list[int]
    n_mines_adjacent: int



class Result(Enum):
    OKAY = 0
    NO_OP = 1
    INVALID = 2
    GAME_OVER = 3


def is_complete(cells: list[Cell]) -> bool:
    return all(cell.revealed for cell in cells if not cell.is_mine)


def unrevealed_adjacent(cell, array):
    return [i for i in cell.adjacent if not (array[i].is_mine and array[i].is_revealed)]


def reveal(loc: int, cells: list[Cell]) -> Result:
    result, revealed, _ = dry_reveal(loc, cells)
    for loc in revealed:
        cells[loc].revealed = True
    return result


def dry_reveal(loc: int, cells: list[Cell]) -> Tuple[Result, set[int], set[int]]:
    cell = cells[loc]
    if cell.revealed:
        return (Result.NO_OP, set(), set())
    if cell.is_mine:
        return (Result.GAME_OVER, set(), set())

    revealed = {cell.loc}
    zeros = set()
    if cell.n_mines_adjacent == 0:
        zeros.add(cell.loc)
        to_reveal = unrevealed_adjacent(cell, cells)
        while to_reveal:
            cell = cells[to_reveal.pop()]
            if cell.loc in revealed:
                continue
            revealed.add(cell.loc)
            if cell.n_mines_adjacent == 0:
                zeros.add(cell.loc)
                to_reveal += unrevealed_adjacent(cell, cells)
    return (Result.OKAY, revealed, zeros)



ALL = 255
TR = 7
RC = 28
BR = 112
LC = 193

DIRS = {
    0: (-1, -1),
    1: (0,  -1),
    2: (1,  -1),
    3: (1,   0),
    4: (1,   1),
    5: (0,   1),
    6: (-1,  1),
    7: (-1,  0),
}



def create_board(n, m, n_mines):
    x = np.random.randn(n *m)
    y = np.partition(x.flatten(), n_mines)[n_mines]
    mine_loc = x < y
    cells: list[Cell] = []
    for i in range(n * m):
        directions = ALL
        directions = directions & ~(TR*(i < n))
        directions = directions & ~(RC*(i % n == n -1))
        directions = directions & ~(BR*(i >= (m-1) * n))
        directions = directions & ~(LC*(i % n == 0))
        adjacent = []
        for j in range(8):
            if directions & 2**j:
                offset_x, offset_y = DIRS[j]
                adjacent.append(i + offset_x + n * offset_y)
        n_mines_adjacent = sum(mine_loc[k] for k in adjacent)
        cell = Cell(False, mine_loc[i], i, adjacent, n_mines_adjacent)
        cells.append(cell)
    return cells


def expert_board():
    return make_board(30, 16, 99)




