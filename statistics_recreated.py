import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn
import statsmodels.api as sm
from scipy import stats
from itertools import groupby
from importer import User, Round
import operator

def regression(rounds):
    simple_regression_dataset=rounds.copy().dropna().groupby("opp_hist").mean()

    cooperation_hist=list(map(lambda x: (100-x)*1./100, simple_regression_dataset.index))

    slope, intercept, r_value, p_value, std_err = stats.linregress(cooperation_hist,simple_regression_dataset["usr_strat"])
    print ("user strat (red): r^2:",r_value**2, "p-value:" ,p_value)
    plt.plot(cooperation_hist, list(map(lambda x: slope*x+intercept,cooperation_hist)),"r-")

    slope, intercept, r_value, p_value, std_err = stats.linregress(cooperation_hist,simple_regression_dataset["opp_strat"])
    print ("opp strat(blue): r^2:",r_value**2, "p-value:" ,p_value)

    plt.plot(cooperation_hist, list(map(lambda x: slope*x+intercept,cooperation_hist)),"b-")

    plt.scatter(cooperation_hist,simple_regression_dataset["usr_strat"] ,  color='red',label="user's average move")
    plt.scatter(cooperation_hist,simple_regression_dataset["opp_strat"] ,  color='blue',label="opponent's average move")

    #plt.plot(simple_regression_dataset.index*1./100, regr.predict(simple_regression_dataset.index), color='red',linewidth=3)
    plt.xlim([-0.01,1.01])
    plt.ylim([0,1])
    plt.xlabel("Opponent cooperation history")
    plt.ylabel("Cooperation rate")
    plt.legend(loc=2)
    plt.show()

def logit_regression_on_usr(rounds):
    logit_regression_data=rounds.copy().dropna()
    logit = sm.Logit(logit_regression_data["usr_strat"],logit_regression_data[["opp_hist","usr_hist"]]*1./100)
    # fit the model
    return logit.fit()

def logit_regression_on_opp(rounds):
    logit_regression_data=rounds.copy().dropna()
    logit = sm.Logit(logit_regression_data["opp_strat"],logit_regression_data[["opp_hist","usr_hist"]]*1./100)
    # fit the model
    return logit.fit()

def plot_logit_regression(rounds):
    result = logit_regression_on_usr(rounds)
    result2 = logit_regression_on_opp(rounds)
    values=np.array([i for i in range(100)])
    values=values*1./100

    plt.plot(values,list(map(lambda x: 1./(1+math.exp(-result.params["opp_hist"]*x-result.params["usr_hist"]*x)),values)),label="responder")
    plt.ylabel("Responder's predicted cooperation rate")
    plt.xlabel("(history) defection rate")

    plt.plot(values,list(map(lambda x: 1./(1+math.exp(-result2.params["opp_hist"]*x-result2.params["usr_hist"]*x)),values)),label="requester")
    plt.ylabel("Predicted cooperation rate")
    plt.xlabel("(history)defection rate")

    plt.legend(loc=3)
    plt.show()

def defection_by_stage(rounds):
    x_values = []
    usr_data = []
    opp_data = []
    for stage, stage_rounds in groupby(sorted(rounds, key=operator.itemgetter(6)), lambda r: r[6]):
        stage_rounds_list = list(stage_rounds)
        opp_rounds = [round for round in stage_rounds_list if round[2]]
        usr_rounds = [round for round in stage_rounds_list if not round[2]]
        opp_defects = len([round for round in opp_rounds if round[0]])
        usr_defects = len([round for round in usr_rounds if round[0]])
        print("In Stage %s, responder defects %d/%d times, or %f of the times" % (stage, opp_defects, len(opp_rounds), opp_defects/len(opp_rounds)))
        print("In Stage %s, initiator defects %d/%d times, or %f of the times" % (stage, usr_defects, len(usr_rounds), usr_defects/len(usr_rounds)))
        x_values.append(stage)
        opp_data.append(opp_defects/len(opp_rounds))
        usr_data.append(usr_defects/len(usr_rounds))
    plot_values("Stage", x_values, usr_data, opp_data)

def defection_by_facebook_friends(rounds):
    x_values = []
    usr_data = []
    opp_data = []
    for is_friend, friend_rounds in groupby(sorted(rounds, key=operator.itemgetter(5)), lambda r: r[5]):
        friends_rounds_list = list(friend_rounds)
        opp_rounds = [round for round in friends_rounds_list if round[2]]
        usr_rounds = [round for round in friends_rounds_list if not round[2]]
        opp_defects = len([round for round in opp_rounds if round[0]])
        usr_defects = len([round for round in usr_rounds if round[0]])
        x_values.append(is_friend==1)
        opp_data.append(opp_defects/len(opp_rounds))
        usr_data.append(usr_defects/len(usr_rounds))
    plot_values("Friends", x_values, usr_data, opp_data)
    
def defection_by_gender(rounds, users):
    x_values = []
    usr_data = {gender:[] for gender in [User.MALE, User.FEMALE, User.UNKNOWN]}
    opp_data = {gender:[] for gender in [User.MALE, User.FEMALE, User.UNKNOWN]}
    for user in users.values():
        usr_games = [round for round in rounds if round.usr==user]
        usr_defects = len([round for round in usr_games if round.usr_strat])
        opp_games = [round for round in rounds if round.opp==user]
        opp_defects = len([round for round in opp_games if round.opp_strat])
        if len(usr_games):
            usr_data[user.gender].append(usr_defects/len(usr_games))
        if len(opp_games):
            opp_data[user.gender].append(opp_defects/len(opp_games))
    for gender in [User.MALE, User.FEMALE]:
        num_bins = 20
        counts, bin_edges = np.histogram(usr_data[gender], bins=num_bins, normed=True)
        cdf = np.cumsum(counts)/num_bins
        plt.plot(bin_edges[1:], cdf, label="%s inviters" % gender)
        counts, bin_edges = np.histogram(opp_data[gender], bins=num_bins, normed=True)
        cdf = np.cumsum(counts)/num_bins
        plt.plot(bin_edges[1:], cdf, label="%s inviteds" % gender)
    plt.legend()
    plt.show()

def defection_by_age(rounds, users):
    x_values = []
    age_ranges = [(0,20),(20,30),(30,40),(50,100)]
    usr_data = {age:[] for age in age_ranges}
    opp_data = {age:[] for age in age_ranges}
    for user in users.values():
        usr_games = [round for round in rounds if round.usr==user]
        usr_defects = len([round for round in usr_games if round.usr_strat])
        opp_games = [round for round in rounds if round.opp==user]
        opp_defects = len([round for round in opp_games if round.opp_strat])
        if len(usr_games) and user.age:
            for age_range in usr_data.keys():
                if user.age>age_range[0] and user.age<age_range[1]:
                    usr_data[age_range].append(usr_defects/len(usr_games))
                    break
        if len(opp_games) and user.age:
            for age_range in opp_data.keys():
                if user.age>age_range[0] and user.age<age_range[1]:
                    opp_data[age_range].append(opp_defects/len(opp_games))
                    break
    for age_range in usr_data.keys():
        bins = np.arange(0,1,.001)
        if len(usr_data[age_range])>0:
            counts, bin_edges = np.histogram(usr_data[age_range], bins=bins, normed=True, density=1)
            cdf = np.cumsum(counts)/len(bins)
            plt.plot(bin_edges[1:], cdf, label="%s inviteds" % (str(age_range[0]) + '-' + str(age_range[1])))
    plt.legend()
    plt.xlabel("Observed defection rate")
    plt.ylabel("Cumulative Distribution")
    plt.show()
    for age_range in opp_data.keys():
        bins = np.arange(0,1,.001)
        if len(opp_data[age_range])>0:
            counts, bin_edges = np.histogram(opp_data[age_range], bins=bins, normed=True, density=1)
            cdf = np.cumsum(counts)/len(bins)
            plt.plot(bin_edges[1:], cdf, label="%s inviters" % (str(age_range[0]) + '-' + str(age_range[1])))
    plt.legend()
    plt.xlabel("Observed defection rate")
    plt.ylabel("Cumulative Distribution")
    plt.show()

def plot_values(x_label, x_values, usr_data, opp_data):
    ax = plt.subplot(111)
    ax.bar(np.arange(len(x_values))-.1, opp_data,width=0.2,color='b',align='center', label='Inviter')
    ax.bar(np.arange(len(x_values))+.1, usr_data,width=0.2,color='r',align='center', label='Invited')
    ax.set_xlabel(x_label)
    ax.set_ylabel("Defection rate")
    plt.xticks(np.arange(len(x_values)),x_values)
    plt.legend()
    plt.show()