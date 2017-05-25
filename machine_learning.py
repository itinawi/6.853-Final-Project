import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
# % matplotlib inline
import sklearn
import statsmodels.api as sm
from scipy import stats


# read the .csv file
#Raw data summarized 
prisoner_data=pd.read_csv('game_data_4600.csv')
del prisoner_data["Unnamed: 0"]
print(prisoner_data.head())


simple_regression_dataset=prisoner_data.copy().dropna().groupby("opp_hist").mean()
#simple_regression_dataset=simple_regression_dataset[simple_regression_dataset["stage"]!=4]

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
plt.legend(loc=2);


########################################################################################################################
########################################################################################################################
########################################################################################################################

logit_regression_data=prisoner_data.copy().dropna()

logit = sm.Logit(logit_regression_data["usr_strat"],logit_regression_data[["opp_hist","usr_hist"]]*1./100)
# fit the model
result = logit.fit()
result.summary2()


########################################################################################################################
########################################################################################################################
########################################################################################################################


logit = sm.Logit(logit_regression_data["opp_strat"],logit_regression_data[["opp_hist","usr_hist"]]*1./100)
# fit the model
result2 = logit.fit()
result2.summary()

########################################################################################################################
########################################################################################################################
############################################


#fig, ax = plt.subplots(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')    
#plt.subplot(221)
values=np.array([i for i in range(100)])
values=values*1./100

#plt.plot(values,list(map(lambda x:logit.predict(x),values)),label="requester")
plt.plot(values,list(map(lambda x: 1./(1+math.exp(-result.params["opp_hist"]*x-result.params["usr_hist"]*x)),values)),label="responder")
#plt.plot(values,list(map(lambda x: 1./(1+math.exp(result.params["usr_hist"]*x)),values)),label="responder")
plt.ylabel("Responder's predicted cooperation rate")
plt.xlabel("(history) defection rate")
#plt.legend(loc=2)
#plt.ylim([0.45,0.65])
#plt.subplot(222)
plt.plot(values,list(map(lambda x: 1./(1+math.exp(-result2.params["opp_hist"]*x-result2.params["usr_hist"]*x)),values)),label="requester")
#plt.plot(values,list(map(lambda x: 1./(1+math.exp(result2.params["usr_hist"]*x)),values)),label="responder")

plt.ylabel("Predicted cooperation rate")
plt.xlabel("(history)defection rate")
plt.legend(loc=3)
#plt.ylim([0.45,0.65]);

########################################################################################################################
########################################################################################################################
########################################################################################################################

X=prisoner_data.copy().dropna()
X=X[X["stage"]!=4]
X["opp_hist"]=X["opp_hist"]*1./100
X["usr_hist"]=X["usr_hist"]*1./100

Y_usr=X["usr_strat"]
Y_opp=X["opp_strat"]
Z=prisoner_data[["CC","CS","SC","SS"]] 
#del X["CS"]
#del X["SS"]
#del X["SC"]
#del X["CC"]

del X["usr_strat"]
del X["opp_strat"]
del X["usr_age"] # seems to be too specific to the users, hence causes overfitting (maybe introduce classes?)
del X["opp_age"]
#del X["stage"] #redundant

########################################################################################################################
########################################################################################################################
########################################################################################################################

from sklearn.decomposition import PCA
from sklearn import preprocessing
import math

pca=PCA(n_components=3)
X_train=X
X_train_centered2 = (X_train-np.mean(X_train,axis=0))*1./np.std(X_train) #normalize data
X_train_centered2=preprocessing.scale(X_train)
X_2d=pca.fit_transform(X_train_centered2)
print ("Explained variance by the first 3 components", pca.explained_variance_ratio_)
print ("Loadings PC1 PC2")
elements=[(X.columns[count],pca.components_[0][count],pca.components_[1][count]) for count in range(len(X.columns))]

for e in reversed(sorted(elements,key=lambda x: x[1])):
    print (str(e).strip("()"))

########################################################################################################################
########################################################################################################################
########################################################################################################################

#Lets draw it

# def helper_drawer(labeling_axis,binary,pcs):
#     #you can use this helper function to draw the projections for any two pcs and column of choice (see examples below)
#     if binary:
#         color = ["blue" if item==1 else "yellow" for item in labeling_axis]
#     else:
#         color=[str(item/255.) for item in labeling_axis]
#     plt.scatter(X_2d[:,pcs[0]],X_2d[:,pcs[1]],c=color,alpha=0.2)
#     plt.xlabel('PC '+str(pcs[0]+1))
#     plt.ylabel('PC '+str(pcs[1]+1))
#     plt.title("PCs "+str(pcs))

# fig, ax = plt.subplots(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')    
# plt.subplot(221)
# helper_drawer(Y_usr,True,(0,1))
# plt.title('First two PCs, yellow means user defected')

# plt.subplot(222)
# helper_drawer(Y_opp,True,(0,1))
# plt.title('First two PCs, yellow means opp defected')

# plt.subplot(223)
# helper_drawer(Y_usr,True,(1,2))
# plt.title('2nd and 3rd PCs, yellow means user defect')

# plt.subplot(224)
# helper_drawer(Y_opp,True,(1,2))
# plt.title('2nd and 3rd PCs, yellow means opp defect')


########################################################################################################################
########################################################################################################################
########################################################################################################################


# fig, ax = plt.subplots(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')    
# plt.subplot(221)
# helper_drawer(X["CC"],False,(0,1))
# plt.title("CC")
# plt.subplot(222)
# helper_drawer(X["SC"],False,(0,1))
# plt.title("SC")
# plt.subplot(223)
# helper_drawer(X["CS"],False,(0,1))
# plt.title("CS")
# plt.subplot(224)
# helper_drawer(X["SS"],False,(0,1))
# plt.title("SS")


########################################################################################################################
########################################################################################################################
########################################################################################################################

# fig, ax = plt.subplots(num=1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')    
# plt.subplot(221)
# helper_drawer(X["usr_hist"],False,(0,1))
# plt.subplot(222)
# helper_drawer(X["opp_hist"],False,(0,1))

########################################################################################################################
########################################################################################################################
########################################################################################################################

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split

#Leave 25% out as test set 
X_train,X_test,Y_train,Y_test=train_test_split(X_2d,Y_usr)
#Do cross_validation
clf = AdaBoostClassifier(n_estimators=200)
scores = cross_val_score(clf, X_train, Y_train)
print ("Predicting usr strategies")
print ("CV score:", np.mean(scores))
#Evaluate on the left-out set
clf.fit(X_train,Y_train)
print ("test score: ",clf.score(X_test,Y_test))
feature_impts=[]
for f in range(len(clf.feature_importances_)):
    feature_impts.append((clf.feature_importances_[f],X.columns[f]))
sorted_f=sorted(feature_impts)
sorted_f.reverse()
print ("Important features: ", sorted_f[:3])



#Leave 25% out as test set 
X_train,X_test,Y_train,Y_test=train_test_split(X_2d,Y_opp)
#Do cross_validation
clf = AdaBoostClassifier(n_estimators=200)
scores = cross_val_score(clf, X_train, Y_train)
print ("Predicting opp strategies")
print ("CV score:", np.mean(scores))
#Evaluate on the left-out set
clf.fit(X_train,Y_train)
print ("test score: ",clf.score(X_test,Y_test))
for f in range(len(clf.feature_importances_)):
    feature_impts.append((clf.feature_importances_[f],X.columns[f]))
sorted_f=sorted(feature_impts)
sorted_f.reverse()
print ("Important features", sorted_f[:3])



########################################################################################################################
########################################################################################################################
########################################################################################################################

# fit a neural network model

