import sys
import csv
from importer import *

def export(dataset):
    with open('exported_rounds.csv', 'w', encoding='utf8',newline='') as csvfile:
        row_writer = csv.writer(csvfile, delimiter=',')
        row_writer.writerow(['strat', 'id','is_requester','age','gender','are_friends','stage', 'mutual_friends', 'hist'] \
            + ['start_%d' % i for i in range(0,5)]
            + ['memory_%d' % i for i in range(0,10)]
            + ['direct_hist_%d' % i for i in range(0,5)])
        for row in dataset:
            row_writer.writerow(row)

def export_moves(moves_dataset):
    with open('exported_moves.csv', 'w', encoding='utf8',newline='') as csvfile:
        row_writer = csv.writer(csvfile, delimiter=',')
        row_writer.writerow(['opp_strat'])
        for row in moves_dataset:
            row_writer.writerow(row)