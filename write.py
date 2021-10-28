#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ProjectName: HW2
# FileName: write
# Description:
# TodoList:
from functools import reduce
import csv


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")


def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"

    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"

    with open(path, 'w') as f:
        f.write(res[:-1])


def writeDataset(piece_type, board, movex, movey, value=0.5):
    flattened_board = reduce(lambda x, y: x + y, board)
    actions = [0 for _ in range(len(flattened_board))]
    actions[len(board[0])*movex + movey] = 1
    row = [flattened_board, actions, value]
    if piece_type == 1:
        with open('black_dataset_new.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            if movex == -1 and movey == -1:
                writer.writerow([])
    else:
        with open('white_dataset_new.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            if movex == -1 and movey == -1:
                writer.writerow([])
    return
