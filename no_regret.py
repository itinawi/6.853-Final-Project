from itertools import groupby
from operator import itemgetter
import numpy as np
import random
import math


def follow_the_leader(dataset):
    sorted_dataset = np.insert(dataset,29,range(len(dataset)),axis=1)
    sorted_dataset = sorted_dataset[np.argsort(sorted_dataset[:, 1], kind='mergesort')]
    grouped_data = {user:np.array([x for x in rounds]) for user, rounds in groupby(sorted_dataset,key = itemgetter(1))}
    predictions = np.empty((0,2))
    for user in grouped_data.keys():
        correct_user_predictions = 0
        #print(user, len(grouped_data[user]),"games, and defected", len(grouped_data[user][np.equal(grouped_data[user][:, 0], 1)])/len(grouped_data[user]))
        games_played = 0
        defections = 0
        for game_round in grouped_data[user]:
            if games_played:
                if defections*2 > games_played:
                    prediction = 1
                elif defections*2 < games_played:
                    prediction = 0
                else:
                    prediction = random.randint(0,1)
            else:
                prediction = random.randint(0,1)
            predictions = np.append(predictions,np.array([[game_round[29],prediction]]), axis=0)
            defections += 1 if game_round[0] else 0
            games_played +=1
    predictions = predictions[np.argsort(predictions[:, 0])][:,1]
    return predictions

def omniscient_follow_the_leader(dataset):
    sorted_dataset = np.insert(dataset,29,range(len(dataset)),axis=1)
    sorted_dataset = sorted_dataset[np.argsort(sorted_dataset[:, 1], kind='mergesort')]
    grouped_data = {user:np.array([x for x in rounds]) for user, rounds in groupby(sorted_dataset,key = itemgetter(1))}
    predictions = np.empty((0,2))
    all_prediction_accuracies = []
    for user in grouped_data.keys():
        total_defection_rate = len(grouped_data[user][np.equal(grouped_data[user][:, 0], 1)])/len(grouped_data[user])
        if total_defection_rate < .5:
            prediction = 0
        elif total_defection_rate > .5:
            prediction = 1
        else:
            prediction = random.randint(0,1)
        for game_round in grouped_data[user]:
            predictions = np.append(predictions,np.array([[game_round[29],prediction]]), axis=0)
    predictions = predictions[np.argsort(predictions[:, 0])][:,1]
    return predictions

def no_regret(dataset):
    sorted_dataset = np.insert(dataset,29,range(len(dataset)),axis=1)
    sorted_dataset = sorted_dataset[np.argsort(sorted_dataset[:, 1], kind='mergesort')]
    grouped_data = {user:np.array([x for x in rounds]) for user, rounds in groupby(sorted_dataset,key = itemgetter(1))}
    correct_predictions = 0
    predictions = np.empty((0,2))
    R = lambda p: p *math.log(p)+(1-p)*math.log(1-p)
    max_abs_R_times_2 = abs(2 * R(0.5))
    for user in grouped_data.keys():
        games_played = 0
        defections = 0
        for game_round in grouped_data[user]:
            if games_played:
                eta = math.sqrt(max_abs_R_times_2/games_played)
                regularized_probability = math.exp(-eta*(games_played - defections))/(math.exp(-eta*defections)+math.exp(-eta*(games_played-defections)))
                prediction = 1 if random.random() < regularized_probability else 0
            else:
                prediction = random.randint(0,1)
            predictions = np.append(predictions,np.array([[game_round[29],prediction]]), axis=0)
            defections += 1 if game_round[0] else 0
            games_played +=1
    predictions = predictions[np.argsort(predictions[:, 0])][:,1]
    return predictions
