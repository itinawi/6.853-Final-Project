import numpy as np

def tit_for_tat(dataset):
    return dataset[:,28].astype(int)

def grim_trigger(Rounds):
    predictions = np.array([])
    for this_round in Rounds:
        if this_round.usr_strat is not None:
            hist = [play for play in [(round.opp_strat if round.usr==this_round.usr else round.usr_strat) for round in Rounds if ((round.usr==this_round.usr and round.opp==this_round.opp) or (round.opp==this_round.usr and round.usr==this_round.opp)) and round.id<this_round.id and round.usr_strat is not None] if play is not None]
            if len(hist)>0:
                prediction = 1 if 1 in hist else 0
            else:
                prediction = 0
            predictions = np.append(predictions,np.array([prediction]))
        if this_round.opp_strat is not None:
            hist = [play for play in [(round.opp_strat if round.usr==this_round.opp else round.usr_strat) for round in Rounds if ((round.usr==this_round.usr and round.opp==this_round.opp) or (round.opp==this_round.usr and round.usr==this_round.opp)) and round.id<this_round.id and round.usr_strat is not None] if play is not None]
            if len(hist)>0:
                prediction = 1 if 1 in hist else 0
            else:
                prediction = 0
            predictions = np.append(predictions,np.array([prediction]))
    return predictions

def pavlov(Rounds):
    predictions = np.array([])
    for this_round in Rounds:
        if this_round.usr_strat is not None:
            hist = [play for play in [(round.opp_strat, round.usr_strat) for round in Rounds if ((round.usr==this_round.usr and round.opp==this_round.opp) or (round.opp==this_round.usr and round.usr==this_round.opp)) and round.id<this_round.id and round.usr_strat is not None] if play[0] is not None and play[1] is not None]
            if len(hist)>0:
                prediction = 0 if hist[-1][0] == hist[-1][1] else 1
            else:
                prediction = 0
            predictions = np.append(predictions,np.array([prediction]))
        if this_round.opp_strat is not None:
            hist = [play for play in [(round.opp_strat, round.usr_strat) for round in Rounds if ((round.usr==this_round.usr and round.opp==this_round.opp) or (round.opp==this_round.usr and round.usr==this_round.opp)) and round.id<this_round.id and round.usr_strat is not None] if play[0] is not None and play[1] is not None]
            if len(hist)>0:
                prediction = 0 if hist[-1][0] == hist[-1][1] else 1
            else:
                prediction = 0
            predictions = np.append(predictions,np.array([prediction]))
    return predictions

def gradual(Rounds): #Code based on https://codegolf.stackexchange.com/a/2422
    predictions = np.array([])
    for this_round in Rounds:
        if this_round.usr_strat is not None:
            hist = [play for play in [(round.opp_strat, round.usr_strat) if round.usr==this_round.usr else (round.usr_strat,round.opp_strat) for round in Rounds if ((round.usr==this_round.usr and round.opp==this_round.opp) or (round.opp==this_round.usr and round.usr==this_round.opp)) and round.id<this_round.id and round.usr_strat is not None] if play[0] is not None and play[1] is not None]
            if len(hist)>0:
                prediction = determine_gradual_play_from_hist(hist)
            else:
                prediction = 0
            predictions = np.append(predictions,np.array([prediction]))
        if this_round.opp_strat is not None:
            hist = [play for play in [(round.opp_strat, round.usr_strat) if round.usr==this_round.opp else (round.usr_strat,round.opp_strat) for round in Rounds if ((round.usr==this_round.usr and round.opp==this_round.opp) or (round.opp==this_round.usr and round.usr==this_round.opp)) and round.id<this_round.id and round.usr_strat is not None] if play[0] is not None and play[1] is not None]
            if len(hist)>0:
                prediction = determine_gradual_play_from_hist(hist)
            else:
                prediction = 0
            predictions = np.append(predictions,np.array([prediction]))
    return predictions

def determine_gradual_play_from_hist(hist):
    current_sequence = 0
    total_defects = 0
    for play in hist:
        if play[0]==1:
            total_defects += 1
            current_sequence = 0
        elif play[1]==0:
            current_sequence += 1
    if current_sequence<total_defects:
        return 1
    elif hist[-1][1]==1 or (len(hist)>1 and hist[-2][1]==1):
        return 0
    elif hist[-1][0]==1 and hist[-1][1]==0:
        return 1
    else:
        return 0