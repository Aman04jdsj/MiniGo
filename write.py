#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ProjectName: HW2
# FileName: write
# Description:
# TodoList:
from functools import reduce
import csv
import pandas as pd


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


def writeDataset(piece_type, board, value=0):
    flattened_board = reduce(lambda x, y: x + y, board)
    row = [flattened_board, value]
    if piece_type == 1:
        with open('black.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    else:
        with open('white.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    return
