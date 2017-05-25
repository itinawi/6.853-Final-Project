from importer import *
from statistics_recreated import *
from exporter import *
from dataset_creator import *
from dataset_reader import *
from svm_learn import *
from no_regret import *
from game_theory_strategies import *
from quantal_response import *


''' Read raw data '''
get_users()
Rounds, raw_data = get_rounds()

''' Read dataset '''
dataset = read_dataset()
opp_moves = read_opp_moves()
print(dataset.shape)
y_values = dataset[:,0].astype(int)
x_values = dataset[:,1:]

''' Stupid Predictions'''
#print('Always guessing 0: ',sum(np.zeros_like(y_values)==y_values)/len(y_values))
#print('Random guessing 50/50:', sum(np.random.randint(0,2,len(y_values))==y_values)/len(y_values))
#
#''' Game Theory Strategies '''
tit_for_tat_predictions = tit_for_tat(dataset)
print('Tit for Tat:', sum(tit_for_tat_predictions==dataset[:,0].astype(int))/len(dataset))
trigger_predictions = grim_trigger(Rounds)
print('Grim Trigger:', sum(trigger_predictions==dataset[:,0].astype(int))/len(dataset))
pavlov_predictions = pavlov(Rounds)
print('Pavlov:', sum(pavlov_predictions==dataset[:,0].astype(int))/len(dataset))
gradual_predictions = gradual(Rounds)
print('Gradual:', sum(gradual_predictions==dataset[:,0].astype(int))/len(dataset))
#
#''' Quantal Response '''
#quantal_response(dataset, .5)
#
#''' Neural Net '''
#
#''' No regret learning '''
#predictions = omniscient_follow_the_leader(dataset)
#print('Omniscient Follow the Leader algorithm:', sum(predictions==dataset[:,0].astype(int))/len(dataset))
follow_leader_predictions = follow_the_leader(dataset)
print('Follow The Leader algorithm:', sum(follow_leader_predictions==dataset[:,0].astype(int))/len(dataset))
no_regret_predictions = no_regret(dataset)
print('No Regret/Follow The Regularized Leader algorithm:', sum(no_regret_predictions==dataset[:,0].astype(int))/len(dataset))

''' Features '''
features = np.empty((len(dataset),6))
features[:,0] = tit_for_tat_predictions
features[:,1] = trigger_predictions
features[:,2] = pavlov_predictions
features[:,3] = gradual_predictions
features[:,4] = follow_leader_predictions
features[:,5] = no_regret_predictions
print(features)
np.savetxt("features.csv",features, fmt='%f', delimiter=',')
#
#''' Learn via SVM'''
#accuracy = svm_learn(x_values, y_values)
#print('SVM algorithm:', accuracy) 
#
#''' Statistics '''
#defection_by_stage(dataset)
#defection_by_facebook_friends(dataset)
#defection_by_gender(Rounds, Users)
#defection_by_age(Rounds, Users)

''' Dataset Creation '''
#dataset = create_dataset(Rounds, Users)
#export(dataset)

''' Opponent Move Creation '''
#moves_dataset = create_opp_moves(Rounds)
#export_moves(moves_dataset)