from importer import *
import numpy as np

def create_dataset(Rounds, Users):
    dataset = []
    for round in Rounds:
        if round.id%1000==0:
            sys.stdout.write('\rExported '+str(round.id))
            sys.stdout.flush()
        if round.usr_strat is not None:
            dataset.append([round.usr_strat, round.usr.id, 0, round.usr.age, 1 if round.usr.gender==User.FEMALE else 0, 1 if round.fb_friend else 0, int(round.stage), round.mutual_friends, round.opp_hist()] + round.usr_start() + round.usr_memory() + round.direct_hist_usr())
        if round.opp_strat is not None:
            dataset.append([round.opp_strat, round.opp.id, 1, round.opp.age, 1 if round.opp.gender==User.FEMALE else 0, 1 if round.fb_friend else 0, int(round.stage), round.mutual_friends, round.usr_hist()] + round.opp_start() + round.opp_memory() + round.direct_hist_opp())
    sys.stdout.write("\rDone Exporting!")
    print()
    return np.array(dataset)


def create_opp_moves(Rounds):
    dataset = []
    for round in Rounds:
        if round.id%1000==0:
            sys.stdout.write('\rExported '+str(round.id))
            sys.stdout.flush()
        if round.usr_strat is not None:
            dataset.append([round.opp_strat if round.opp_strat != None else .5])
        if round.opp_strat is not None:
            dataset.append([round.usr_strat if round.usr_strat != None else .5])
    sys.stdout.write("\rDone Exporting!")
    print()
    return np.array(dataset)