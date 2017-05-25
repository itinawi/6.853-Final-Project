import sys
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

Users = {}
Rounds = []
Stages = {
    '2':{
        'CC':3,
        'CS':-2,
        'SC':5,
        'SS':0
    },
    '3':{
        'CC':6,
        'CS':-2,
        'SC':8,
        'SS':0
    },
    '4':{
        'CC':6,
        'CS':0,
        'SC':12,
        'SS':-12
    },
    '5':{
        'CC':12,
        'CS':-8,
        'SC':20,
        'SS':0
    },
    '6':{
        'CC':12,
        'CS':-4,
        'SC':16,
        'SS':0
    }
}
boolDict = {'t':True,'f':False}

class User:
    UNKNOWN = "Unknown"
    MALE = "Male"
    FEMALE = "Female"
    def __init__(self, id, birth_date, gender):
        self.id = id
        if birth_date==birth_date: #Checking for nan
            birthdate = datetime.strptime(birth_date,"%m/%d/%Y")
            self.age = relativedelta(datetime(2010,1,1), birthdate).years
        else:
            self.age = 0
        if gender=="male":
            self.gender = User.MALE
        elif gender=="female":
            self.gender = User.FEMALE
        else:
            self.gender = User.UNKNOWN
    def to_row(self):
        return [self.id, self.age, self.gender]

class Round:
    def __init__(self, id, usr_id, opp_id, fb_friend, stage, usr_strat, opp_strat, mutual_friends):
        self.id = id
        if usr_id not in Users:
            Users[usr_id]= User(usr_id, float('nan'),'')
        self.usr = Users[usr_id]
        self.opp = Users[opp_id]
        self.usr_strat = (1 if boolDict[usr_strat] else 0) if usr_strat==usr_strat else None
        self.opp_strat = (1 if boolDict[opp_strat] else 0) if opp_strat==opp_strat else None
        self.fb_friend = boolDict[fb_friend] if fb_friend==fb_friend else False
        self.mutual_friends = int(mutual_friends) if mutual_friends==mutual_friends else 0
        self.stage = str(int(stage))
    def usr_start(self):
        start = [play for play in [(round.opp_strat if round.usr==self.usr else round.usr_strat) for round in Rounds if (round.usr==self.usr or round.opp==self.usr) and round.id<self.id and round.usr_strat is not None] if play is not None][:5]
        return start + [.5]*(5-len(start))
    def opp_start(self):
        start = [play for play in [(round.opp_strat if round.usr==self.opp else round.usr_strat) for round in Rounds if (round.usr==self.opp or round.opp==self.opp) and round.id<self.id and round.usr_strat is not None] if play is not None][:5]
        return start + [.5]*(5-len(start))
    def usr_memory(self):
        mem = [play for play in [(round.opp_strat if round.usr==self.usr else round.usr_strat) for round in Rounds if (round.usr==self.usr or round.opp==self.usr) and round.id<self.id and round.usr_strat is not None] if play is not None][-10:]
        return [.5]*(10-len(mem)) + mem
    def opp_memory(self):
        mem = [play for play in [(round.opp_strat if round.usr==self.opp else round.usr_strat) for round in Rounds if (round.usr==self.opp or round.opp==self.opp) and round.id<self.id and round.usr_strat is not None] if play is not None][-10:]
        return [.5]*(10-len(mem)) + mem
    def usr_hist(self):
        return sum([play for play in [(round.usr_strat if round.usr==self.usr else round.opp_strat) for round in Rounds if (round.usr==self.usr or round.opp==self.usr) and round.id<self.id and round.usr_strat is not None] if play is not None][-5:])*20
    def opp_hist(self):
        return sum([play for play in [(round.usr_strat if round.usr==self.opp else round.opp_strat) for round in Rounds if (round.usr==self.opp or round.opp==self.opp) and round.id<self.id and round.usr_strat is not None] if play is not None][-5:])*20
    def direct_hist_usr(self):
        hist = [play for play in [(round.opp_strat if round.usr==self.usr else round.usr_strat) for round in Rounds if ((round.usr==self.usr and round.opp==self.opp) or (round.opp==self.usr and round.usr==self.opp)) and round.id<self.id and round.usr_strat is not None] if play is not None][-5:]
        return [.5]*(5-len(hist)) + hist
    def direct_hist_opp(self):
        hist = [play for play in [(round.opp_strat if round.usr==self.opp else round.usr_strat) for round in Rounds if ((round.usr==self.usr and round.opp==self.opp) or (round.opp==self.usr and round.usr==self.opp)) and round.id<self.id and round.usr_strat is not None] if play is not None][-5:]
        return [.5]*(5-len(hist)) + hist
    def to_row(self):
        return [self.id, self.usr.id, self.opp.id, self.fb_friend, self.stage, self.usr_strat, self.opp_strat, self.mutual_friends, self.usr_hist(), self.opp_hist()] \
        + self.usr_start() \
        + self.opp_start() \
        + self.usr_memory() \
        + self.opp_memory() \
        + self.direct_hist_usr() \
        + self.direct_hist_opp()

def get_users():
    user_data=pd.read_csv('users_raw_anon.csv')
    del user_data["Unnamed: 0"]
    user_data.head()

    for row in range(len(user_data)):
        userDict = user_data.ix[row].to_dict()
        user = User(userDict['id'],userDict['birth_date'], userDict['gender'])
        Users[user.id] = user


def get_rounds():
    rounds_data=pd.read_csv('games_raw_anon.csv')
    del rounds_data["Unnamed: 0"]
    rounds_data.head()
    data_length = len(rounds_data)
    #data_length = 100
    for row in range(data_length):
        if row%1000==0:
            sys.stdout.write('\rReading row '+str(row))
            sys.stdout.flush()
        roundDict = rounds_data.ix[row].to_dict()
        if roundDict['stage_id'] != 1:
            round = Round(row, roundDict['user_id'],roundDict['opp_id'],roundDict['fb_friend'],roundDict['stage_id'],roundDict['user_strat'],roundDict['opp_strat'],roundDict['mutual_friends'],)
            Rounds.append(round)
    sys.stdout.write("\rLoaded!")
    print()
    return (Rounds, rounds_data)