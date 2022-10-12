# -*- coding: utf-8 -*-

# Paper Title: A novel control factor and Brownian motion‑based improved Harris Hawks Optimization for feature selection
# Link to paper: https://link.springer.com/content/pdf/10.1007/s12652-021-03621-y.pdf

from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import classification_report,accuracy_score,mean_squared_error,log_loss
from sklearn.preprocessing import Binarizer
from sklearn import metrics
from math import pi,gamma,sin
from scipy.special import erf
import random
import math
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm_notebook
from copy import deepcopy
from numpy.matlib import repmat
from collections import Counter
import operator
import warnings
warnings.filterwarnings('ignore')

# loading dataset from google drive
 
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def initialization(datainfo):

  if datainfo=='OSCC':
    link = 'https://drive.google.com/open?id=1IR4QBufB9bEr3q416tCjeqyEYsHIXHtP'
    fluff, id = link.split('=')
    downloaded = drive.CreateFile({'id':id}) 
    downloaded.GetContentFile('OSCC.csv')  
    data = pd.read_csv('OSCC.csv')
    data.rename(columns={'Target':'class'},inplace=True)

  if datainfo=='Breast':
    link = 'https://drive.google.com/open?id=13wC7XKjaMgzwqNuF2bqL1w_86Q7-2POU'
    fluff, id = link.split('=')
    downloaded = drive.CreateFile({'id':id}) 
    downloaded.GetContentFile('Breast.csv')  
    data = pd.read_csv('Breast.csv')

  if datainfo=='Colon':
    link = 'https://drive.google.com/open?id=1Zp9rrDxZMPf9ZIQp8KoAlzTaiqea-Qms'
    fluff, id = link.split('=')
    downloaded = drive.CreateFile({'id':id}) 
    downloaded.GetContentFile('Colon.csv')  
    data = pd.read_csv('Colon.csv')

  if datainfo=='Leukemia':
    link = 'https://drive.google.com/open?id=1nqcnbKOhWYg6mmLgF8wrytabTmB09tJy'
    fluff, id = link.split('=')
    downloaded = drive.CreateFile({'id':id}) 
    downloaded.GetContentFile('Leukemia.csv')  
    data = pd.read_csv('Leukemia.csv')

  if datainfo=='Ovarian':
    link = 'https://drive.google.com/open?id=1SBdxkrcJ05VTgytcJqohiSBG7Do2-t39'
    fluff, id = link.split('=')
    downloaded = drive.CreateFile({'id':id}) 
    downloaded.GetContentFile('Ovarian.csv')  
    data = pd.read_csv('Ovarian.csv')

  if datainfo=='CNS':
    link = 'https://drive.google.com/open?id=15WlWy2AwwqGQvlhr6JDQb8WHhJNVphoR'
    fluff, id = link.split('=')
    downloaded = drive.CreateFile({'id':id}) 
    downloaded.GetContentFile('CNS.csv')  
    data = pd.read_csv('CNS.csv')

  X = data.drop('class', axis=1)
  Y = data['class']
  # handeling imbalance dataset
  smt = SMOTETomek(random_state=50,ratio='auto')
  X_bal, Y_bal = smt.fit_sample(X, Y)
  # tarin test split
  X_train, X_val, Y_train, Y_val = train_test_split(X_bal, Y_bal, test_size=0.3, stratify = Y_bal, random_state=None)
  # Data Standardisation
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
  return (X_train,X_val,Y_train,Y_val,X,data)

def error(m,n):
  Y_predF = []
  for k in range(0, m.shape[0]):
    perceptron = np.dot(m[k],n) + Bias
    sigmoid = 1.0/(1.0 + np.exp(-perceptron))
    Y_predF.append(sigmoid)
  return log_loss(Y_train,Y_predF)
  
def predict(X):
  Y_pred_new = []
  for x in X:
    perceptron = np.dot(x,Leader_pos) + Bias
    y_pred = 1.0/(1.0 + np.exp(-perceptron))
    # y_pred = np.where(y_pred == 1.0, 1, 0)
    Y_pred_new.append(y_pred)
  return np.array(Y_pred_new)

def Levy(dim):
  beta=1.5
  sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
  u = 0.01*np.random.randn(dim)*sigma
  v = np.random.randn(dim)
  zz = np.power(np.absolute(v),(1/beta))
  step = np.divide(u,zz)
  return step

def levy(n,m,beta):
  num = gamma(1+beta)*sin(pi*beta/2) # used for Numerator 
  den = gamma((1+beta)/2)*beta*2**((beta-1)/2) # used for Denominator
  sigma_u = (num/den)**(1/beta) # Standard deviation
  u = np.random.normal(loc=0,scale=sigma_u,size=(n,m)) 
  v = np.random.normal(loc=0,scale=1,size=(n,m))
  z = np.true_divide(u,np.power(abs(v),(1/beta)))
  return(z)

def Brown(v):
  # v = np.random.randn(dim)
  step = 1/np.sqrt(2*np.pi)*np.exp(-v**2/2)
  return step

def imp_features(X, Positions):
  col_name = []
  for col in X.columns: 
    col_name.append(col)
  col_name = np.asarray(col_name)
  # series of features and their corresponding weights
  mapped_wt = pd.Series(data=np.mean(Positions,axis=0),index=col_name)
  # taking top n features
  M1 = mapped_wt.nlargest(n=5,keep='first')
  # taking least n features
  M2 = mapped_wt.nsmallest(n=5,keep='first')
  # create single series
  s1 = M1.append(M2)
  # select features from original dataset using indices of s1
  selected_feature_subset = X[s1.index]
  selected = selected_feature_subset.columns
  selected_list = selected.tolist()
  return np.asarray(selected_list)

def plot_roc_curve(fpr, tpr, auc, dataset):
  roc_auc = auc
  plt.plot(fpr, tpr, color='orange', label = 'AUC = %0.2f' % roc_auc)
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(dataset)
  plt.legend()
  plt.show()

"""# Improved Harris Hawk"""

for dataset in ['OSCC','Breast','Colon','Leukemia','Ovarian','CNS']:
  X_train,X_val,Y_train,Y_val,X,data = initialization(dataset)
  SearchAgents_no = 50
  dim = X_train.shape[1]  
  Leader_pos = np.zeros(dim) 
  Leader_score = math.inf
  Max_iter = 100
  stepsize = np.zeros((SearchAgents_no,dim))
  Convergence_curve = np.zeros(Max_iter)
  Train_Accuracy = np.zeros(Max_iter)
  Test_Accuracy = np.zeros(Max_iter)
  Bias = 1
  ub = np.max(X_train,axis=0)
  lb = np.min(X_train,axis=0)
  Positions = np.zeros((SearchAgents_no,dim))
  i=0
  while i < SearchAgents_no:
    for j in range(0,dim):
      Positions[i][j] = ub[j]-(ub[j]-lb[j])*np.random.rand()
    i+=1
  Sel_features = []

  for i in range(0, SearchAgents_no):

    Y_pred = []
    
    for j in range(0, X_train.shape[0]):
      perceptron = np.dot(X_train[j],Positions[i]) + Bias
      sigmoid = 1.0/(1.0 + np.exp(-perceptron))
      Y_pred.append(sigmoid)
    
    fitness = log_loss(Y_train,Y_pred)
    if fitness<Leader_score:
      Leader_score=fitness
      Leader_pos=Positions[i,:].copy() 

  for t in tqdm_notebook(range(Max_iter), total=Max_iter, unit="iterations"):

    # E1=(1-(t/Max_iter))**(2*(t/Max_iter)) # factor to show the decreaing energy of rabbit 
    E1 = 2 * (1 - (t + 1) * 1.0 / Max_iter)       # factor to show the decreasing energy of rabbit

    for i in range(0, SearchAgents_no):
      E0=2*random.random()-1  # -1<E0<1
      Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper

      # Escaping_Energy=E1  # New parameter 

      # -------- Exploration phase Eq. (1) in paper -------------------
      if abs(Escaping_Energy)>=1:
        # Harris' hawks perch randomly based on 2 strategy:
        q = random.random()
        rand_Hawk_index = math.floor(SearchAgents_no*random.random())
        X_rand = Positions[rand_Hawk_index, :]
        if q>=0.5:
          # perch based on other family members
          Positions[i,:]=X_rand-random.random()*abs(X_rand-2*random.random()*Positions[i,:])

        elif q<0.5:
          #perch on a random tall tree (random site inside group's home range)
          Positions[i,:]=(Leader_pos - Positions.mean(0))-random.random()*((ub-lb)*random.random()+lb)

        # ----------Exploitation phase-------------------------------------
        elif abs(Escaping_Energy)<1:
          # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
          # phase 1: ----- surprise pounce (seven kills) ----------
          # surprise pounce (seven kills): multiple, short rapid dives by different hawks

          r=random.random() # probablity of each event
                
          if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
            Positions[i,:]=(Leader_pos)-Escaping_Energy*abs(Leader_pos-Positions[i,:])

          if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
            Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
            Positions[i,:]=(Leader_pos-Positions[i,:])-Escaping_Energy*abs(Jump_strength*Leader_pos-Positions[i,:])

          if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
            #rabbit try to escape by many zigzag deceptive motions
            Jump_strength=2*(1-random.random())
            X1=Leader_pos-Escaping_Energy*abs(Jump_strength*Leader_pos-Positions[i,:])
            X1 = np.clip(X1, lb, ub)
            if error(X_train,X1) < Leader_score: # improved move?
              Positions[i,:] = X1.copy()
            else: # hawks perform levy-based short rapid dives around the rabbit
              # X2=X1+np.multiply(np.random.randn(dim),Levy(dim))
              stepsize = np.multiply(Brown(X1),(Leader_pos-np.multiply(Brown(X1),Positions[i])))
              X2=Positions[i]+0.5*np.multiply(np.random.rand(Positions.shape[1]),stepsize)
              # X2=X1+np.multiply(np.random.randn(dim),Brown(dim))
              # X2 = numpy.clip(X2, lb, ub)
              if error(X_train,X2) < Leader_score:
                Positions[i,:] = X2.copy()
          
          if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
            Jump_strength=2*(1-random.random())
            X1=Leader_pos-Escaping_Energy*abs(Jump_strength*Leader_pos-Positions.mean(0))
            X1 = np.clip(X1, lb, ub)
            if error(X_train,X1) < Leader_score:  # improved move?
              Positions[i,:] = X1.copy()
            else: # Perform levy-based short rapid dives around the rabbit
              # X2=X1+np.multiply(np.random.randn(dim),Levy(dim))
              stepsize = np.multiply(Brown(X1),(Leader_pos-np.multiply(Brown(X1),Positions[i])))
              X2=Positions[i]+0.5*np.multiply(np.random.rand(Positions.shape[1]),stepsize)
              # X2=X1+np.multiply(np.random.randn(dim),Brown(dim))
              # X2 = np.clip(X2, lb, ub)
              if error(X_train,X2) < Leader_score:
                Positions[i,:] = X2.copy()
          
    for i in range(0, SearchAgents_no):
      Y_pred = []
      for j in range(0, X_train.shape[0]):
        perceptron = np.dot(X_train[j],Positions[i]) + Bias
        sigmoid = 1.0/(1.0 + np.exp(-perceptron))
        Y_pred.append(sigmoid)
      fitness = log_loss(Y_train,Y_pred)
      # Update the leader
      if fitness<Leader_score: # Change this to > for maximization problem
        Leader_score=fitness; # Update alpha
        Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position

    Convergence_curve[t]=Leader_score
    # Leader_pos = Positions[i,:].copy()

    sel_features = imp_features(X,Positions)
    Sel_features.append(sel_features)

    Y_train_pred = predict(X_train)
    acc_train = float(mean_squared_error(Y_train, Y_train_pred))
    Train_Accuracy[t]=1-acc_train
   

    Y_test_pred = predict(X_val)
    acc_test = float(mean_squared_error(Y_val, Y_test_pred))
    Test_Accuracy[t]=1-acc_test

    t=t+1

  Results = pd.DataFrame()
  Results['Conv_curve']=Convergence_curve
  Results['Train_acc']=Train_Accuracy
  Results['Test_acc']=Test_Accuracy
  Results.to_csv('iHHO_Conv_Train_Test_of_{}.csv'.format(dataset),header=True,index=False)

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Convergence_curve, label='Convergence Curve')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Fitness')
  plt.show()

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Train_Accuracy,label='Train Accuracy')
  plt.plot(Test_Accuracy, label='Test Accuracy')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  # performance evaluation of the selected feature subset
  features = np.concatenate(Sel_features, axis=0)
  asd = Counter(features)
  sorted_features = sorted(asd.items(), key=operator.itemgetter(1))
  high = asd.most_common(10)
  df = pd.DataFrame(high, columns=['Genes', 'Count'])
  df.to_csv('iHHO_Selected_feature_of_{}.csv'.format(dataset),header=True,index=False)
  f1 = []
  for i in range(0,len(high)):
    f1.append(high[i][0])
  f1.append('class')
  data_new = data[f1]
  X_new = data_new.drop('class', axis=1)
  Y_new = data_new['class']
  smt = SMOTETomek(random_state=50,sampling_strategy='auto')
  X_bal_new, Y_bal_new = smt.fit_sample(X_new, Y_new)
  X_train_new, X_val_new, Y_train_new, Y_val_new = train_test_split(X_bal_new, Y_bal_new, test_size=0.3, stratify = Y_bal_new, random_state=None)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val = model.predict_proba(X_val_new)
  y_val1 = y_val[:,1]
  auc = roc_auc_score(Y_val_new, y_val1)
  fpr, tpr, thresholds = roc_curve(Y_val_new,y_val1)
  plot_roc_curve(fpr, tpr, auc, dataset)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val2 = model.predict(X_val_new)
  report = classification_report(Y_val_new,y_val2, output_dict=True)
  df1 = pd.DataFrame(report).transpose()
  df1.to_csv('iHHO_Classification_report_of_selected_feature_subset_for_{}.csv'.format(dataset),header=True,index=True)

"""# Marine Predator Algorithm"""

for dataset in ['OSCC','Breast','Colon','Leukemia','Ovarian','CNS']:
  X_train,X_val,Y_train,Y_val,X,data = initialization(dataset)
  SearchAgents_no = 50
  dim = X_train.shape[1]  
  Leader_pos = np.zeros(dim) 
  Leader_score = math.inf
  Max_iter = 100
  stepsize = np.zeros((SearchAgents_no,dim))
  Convergence_curve = np.zeros(Max_iter)
  Train_Accuracy = np.zeros(Max_iter)
  Test_Accuracy = np.zeros(Max_iter)
  Bias = 1
  ub = np.max(X_train,axis=0)
  lb = np.min(X_train,axis=0)
  Positions = np.zeros((SearchAgents_no,dim))
  i=0
  while i < SearchAgents_no:
    for j in range(0,dim):
      Positions[i][j] = ub[j]-(ub[j]-lb[j])*np.random.rand()
    i+=1
  Sel_features = []

  for i in range(0, SearchAgents_no):

    Y_pred = []
    
    for j in range(0, X_train.shape[0]):
      perceptron = np.dot(X_train[j],Positions[i]) + Bias
      sigmoid = 1.0/(1.0 + np.exp(-perceptron))
      Y_pred.append(sigmoid)
    
    fitness = log_loss(Y_train,Y_pred)
    if fitness<Leader_score:
      Leader_score=fitness
      Leader_pos=Positions[i,:].copy() 

  FADs=0.2
  P=0.5

  for t in tqdm_notebook(range(Max_iter), total=Max_iter, unit="iterations"):

    #------------------- Marine Memory saving ------------------- 
    
    Elite=repmat(Leader_pos,SearchAgents_no,1)  #(Eq. 10) 
    CF=(1-t/Max_iter)**(2*t/Max_iter)
                              
    RL=0.05*levy(SearchAgents_no,dim,1.5)   #Levy random number vector
    RB=np.random.randn(SearchAgents_no,dim)          #Brownian random number vector
    
    for i in range(0,SearchAgents_no):

      R=np.random.rand()
      #------------------ Phase 1 (Eq.12) ------------------- 
      if t<(Max_iter/3):
        stepsize[i]=RB[i]*(Elite[i]-RB[i]*Positions[i])                    
        Positions[i]=Positions[i]+P*R*stepsize[i]
      #--------------- Phase 2 (Eqs. 13 & 14)----------------
      elif t>(Max_iter/3) and t<(2*Max_iter/3):
        if i>Positions.shape[0]/2:
          stepsize[i]=RB[i]*(RB[i]*Elite[i]-Positions[i])
          Positions[i]=Elite[i]+P*CF*stepsize[i] 
        else:
          stepsize[i]=RL[i]*(Elite[i]-RL[i]*Positions[i])                     
          Positions[i]=Positions[i]+P*R*stepsize[i] 
      #----------------- Phase 3 (Eq. 15)-------------------
      else:   
        stepsize[i]=RL[i]*(RL[i]*Elite[i]-Positions[i])
        Positions[i]=Elite[i]+P*CF*stepsize[i]   
      
    #------------------ Detecting top predator ------------------        
    for i in range(0, SearchAgents_no):
      Y_pred = []
      for j in range(0, X_train.shape[0]):
        perceptron = np.dot(X_train[j],Positions[i]) + Bias
        sigmoid = 1.0/(1.0 + np.exp(-perceptron))
        Y_pred.append(sigmoid)

      fitness = log_loss(Y_train,Y_pred)
    if fitness<Leader_score: # Change this to > for maximization problem
      Leader_score=fitness # Update alpha
      Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position                                        
    
    Convergence_curve[t]=Leader_score

    sel_features = imp_features(X,Positions)
    Sel_features.append(sel_features)

    Y_train_pred = predict(X_train)
    acc_train = float(mean_squared_error(Y_train, Y_train_pred))
    Train_Accuracy[t]=1-acc_train

    Y_test_pred = predict(X_val)
    acc_test = float(mean_squared_error(Y_val, Y_test_pred))
    Test_Accuracy[t]=1-acc_test

    t=t+1

  Results = pd.DataFrame()
  Results['Conv_curve']=Convergence_curve
  Results['Train_acc']=Train_Accuracy
  Results['Test_acc']=Test_Accuracy
  Results.to_csv('MPA_Conv_Train_Test_of_{}.csv'.format(dataset),header=True,index=False)

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Convergence_curve, label='Convergence Curve')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Fitness')
  plt.show()

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Train_Accuracy,label='Train Accuracy')
  plt.plot(Test_Accuracy, label='Test Accuracy')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  # performance evaluation of the selected feature subset
  features = np.concatenate(Sel_features, axis=0)
  asd = Counter(features)
  sorted_features = sorted(asd.items(), key=operator.itemgetter(1))
  high = asd.most_common(10)
  df = pd.DataFrame(high, columns=['Genes', 'Count'])
  df.to_csv('MPA_Selected_feature_of_{}.csv'.format(dataset),header=True,index=False)
  f1 = []
  for i in range(0,len(high)):
    f1.append(high[i][0])
  f1.append('class')
  data_new = data[f1]
  X_new = data_new.drop('class', axis=1)
  Y_new = data_new['class']
  smt = SMOTETomek(random_state=50,sampling_strategy='auto')
  X_bal_new, Y_bal_new = smt.fit_sample(X_new, Y_new)
  X_train_new, X_val_new, Y_train_new, Y_val_new = train_test_split(X_bal_new, Y_bal_new, test_size=0.3, stratify = Y_bal_new, random_state=None)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val = model.predict_proba(X_val_new)
  y_val1 = y_val[:,1]
  auc = roc_auc_score(Y_val_new, y_val1)
  fpr, tpr, thresholds = roc_curve(Y_val_new,y_val1)
  plot_roc_curve(fpr, tpr, auc, dataset)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val2 = model.predict(X_val_new)
  report = classification_report(Y_val_new,y_val2, output_dict=True)
  df1 = pd.DataFrame(report).transpose()
  df1.to_csv('MPA_Classification_report_of_selected_feature_subset_for_{}.csv'.format(dataset),header=True,index=True)

"""# Salp Swarm Optimization"""

for dataset in ['OSCC','Breast','Colon','Leukemia','Ovarian','CNS']:
  X_train,X_val,Y_train,Y_val,X,data = initialization(dataset)
  SearchAgents_no = 50
  dim = X_train.shape[1]  
  Leader_pos = np.zeros(dim) 
  Leader_score = math.inf
  Max_iter = 100
  stepsize = np.zeros((SearchAgents_no,dim))
  Convergence_curve = np.zeros(Max_iter)
  Train_Accuracy = np.zeros(Max_iter)
  Test_Accuracy = np.zeros(Max_iter)
  Bias = 1
  ub = np.max(X_train,axis=0)
  lb = np.min(X_train,axis=0)
  Positions = np.zeros((SearchAgents_no,dim))
  i=0
  while i < SearchAgents_no:
    for j in range(0,dim):
      Positions[i][j] = ub[j]-(ub[j]-lb[j])*np.random.rand()
    i+=1
  Sel_features = []

  for i in range(0, SearchAgents_no):

    Y_pred = []
    
    for j in range(0, X_train.shape[0]):
      perceptron = np.dot(X_train[j],Positions[i]) + Bias
      sigmoid = 1.0/(1.0 + np.exp(-perceptron))
      Y_pred.append(sigmoid)
    
    fitness = log_loss(Y_train,Y_pred)
    if fitness<Leader_score:
      Leader_score=fitness
      Leader_pos=Positions[i,:].copy() 

  for t in tqdm_notebook(range(Max_iter), total=Max_iter, unit="iterations"):

    c1 = 2*np.exp(-(4*t/Max_iter)**2) # Eq. (3.2) in the paper

    for i in range(0, SearchAgents_no):

      c2 = np.random.rand()                 
      c3 = np.random.rand()

      if (i==0):
        if (c3 >= 0.5):        
          Positions[i] = Leader_pos + c1*((ub-lb)*c2+lb)
        else:         
          Positions[i] = Leader_pos - c1*((ub-lb)*c2+lb)

      else:
        Positions[i] = (Positions[i]+Positions[i-1])/2
          
    for i in range(0, SearchAgents_no):
      Y_pred = []
      for j in range(0, X_train.shape[0]):
        perceptron = np.dot(X_train[j],Positions[i]) + Bias
        sigmoid = 1.0/(1.0 + np.exp(-perceptron))
        Y_pred.append(sigmoid)
      fitness = log_loss(Y_train,Y_pred)
      # Update the leader
      if fitness<Leader_score: # Change this to > for maximization problem
        Leader_score=fitness # Update alpha
        Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position

    Convergence_curve[t]=Leader_score

    sel_features = imp_features(X,Positions)
    Sel_features.append(sel_features)

    Y_train_pred = predict(X_train)
    acc_train = float(mean_squared_error(Y_train, Y_train_pred))
    Train_Accuracy[t]=1-acc_train

    Y_test_pred = predict(X_val)
    acc_test = float(mean_squared_error(Y_val, Y_test_pred))
    Test_Accuracy[t]=1-acc_test

    t=t+1

  Results = pd.DataFrame()
  Results['Conv_curve']=Convergence_curve
  Results['Train_acc']=Train_Accuracy
  Results['Test_acc']=Test_Accuracy
  Results.to_csv('SSO_Conv_Train_Test_of_{}.csv'.format(dataset),header=True,index=False)

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Convergence_curve, label='Convergence Curve')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Fitness')
  plt.show()

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Train_Accuracy,label='Train Accuracy')
  plt.plot(Test_Accuracy, label='Test Accuracy')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  # performance evaluation of the selected feature subset
  features = np.concatenate(Sel_features, axis=0)
  asd = Counter(features)
  sorted_features = sorted(asd.items(), key=operator.itemgetter(1))
  high = asd.most_common(10)
  df = pd.DataFrame(high, columns=['Genes', 'Count'])
  df.to_csv('SSO_Selected_feature_of_{}.csv'.format(dataset),header=True,index=False)
  f1 = []
  for i in range(0,len(high)):
    f1.append(high[i][0])
  f1.append('class')
  data_new = data[f1]
  X_new = data_new.drop('class', axis=1)
  Y_new = data_new['class']
  smt = SMOTETomek(random_state=50,sampling_strategy='auto')
  X_bal_new, Y_bal_new = smt.fit_sample(X_new, Y_new)
  X_train_new, X_val_new, Y_train_new, Y_val_new = train_test_split(X_bal_new, Y_bal_new, test_size=0.3, stratify = Y_bal_new, random_state=None)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val = model.predict_proba(X_val_new)
  y_val1 = y_val[:,1]
  auc = roc_auc_score(Y_val_new, y_val1)
  fpr, tpr, thresholds = roc_curve(Y_val_new,y_val1)
  plot_roc_curve(fpr, tpr, auc, dataset)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val2 = model.predict(X_val_new)
  report = classification_report(Y_val_new,y_val2, output_dict=True)
  df1 = pd.DataFrame(report).transpose()
  df1.to_csv('SSO_Classification_report_of_selected_feature_subset_for_{}.csv'.format(dataset),header=True,index=True)

"""# Whale Optimization Algorithm"""

for dataset in ['OSCC','Breast','Colon','Leukemia','Ovarian','CNS']:
  X_train,X_val,Y_train,Y_val,X,data = initialization(dataset)
  SearchAgents_no = 50
  dim = X_train.shape[1]  
  Leader_pos = np.zeros(dim) 
  Leader_score = math.inf
  Max_iter = 100
  stepsize = np.zeros((SearchAgents_no,dim))
  Convergence_curve = np.zeros(Max_iter)
  Train_Accuracy = np.zeros(Max_iter)
  Test_Accuracy = np.zeros(Max_iter)
  Bias = 1
  ub = np.max(X_train,axis=0)
  lb = np.min(X_train,axis=0)
  Positions = np.zeros((SearchAgents_no,dim))
  i=0
  while i < SearchAgents_no:
    for j in range(0,dim):
      Positions[i][j] = ub[j]-(ub[j]-lb[j])*np.random.rand()
    i+=1
  Sel_features = []

  for i in range(0, SearchAgents_no):

    Y_pred = []
    
    for j in range(0, X_train.shape[0]):
      perceptron = np.dot(X_train[j],Positions[i]) + Bias
      sigmoid = 1.0/(1.0 + np.exp(-perceptron))
      Y_pred.append(sigmoid)
    
    fitness = log_loss(Y_train,Y_pred)
    if fitness<Leader_score:
      Leader_score=fitness
      Leader_pos=Positions[i,:].copy() 

  for t in tqdm_notebook(range(Max_iter), total=Max_iter, unit="iterations"):

    a=2-t*((2)/Max_iter)
  
    a2=-1+t*((-1)/Max_iter)
    
    for i in range(0,SearchAgents_no):

      r1=random.random()
      r2=random.random()
      
      A=2*a*r1-a 
      C=2*r2   

      b=1            
      l=(a2-1)*random.random()+1   
      
      p = random.random()      
      
      if p<0.5:
        if abs(A)>=1:
          rand_leader_index = math.floor(SearchAgents_no*random.random())
          X_rand = Positions[rand_leader_index]
          D_X_rand=abs(C*X_rand-Positions[i]) 
          Positions[i]=X_rand-A*D_X_rand   #update statement

        elif abs(A)<1:
          D_Leader=abs(C*Leader_pos-Positions[i]) 
          Positions[i]=Leader_pos-A*D_Leader    #update statement  

      elif p>=0.5:
        distance2Leader=abs(Leader_pos-Positions[i])
        Positions[i]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos
            
    for i in range(0, SearchAgents_no):
      Y_pred = []
      for j in range(0, X_train.shape[0]):
        perceptron = np.dot(X_train[j],Positions[i]) + Bias
        sigmoid = 1.0/(1.0 + np.exp(-perceptron))
        Y_pred.append(sigmoid)
      fitness = log_loss(Y_train,Y_pred)
      # Update the leader
      if fitness<Leader_score: # Change this to > for maximization problem
        Leader_score=fitness # Update alpha
        Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position

    Convergence_curve[t]=Leader_score

    sel_features = imp_features(X,Positions)
    Sel_features.append(sel_features)

    Y_train_pred = predict(X_train)
    acc_train = float(mean_squared_error(Y_train, Y_train_pred))
    Train_Accuracy[t]=1-acc_train

    Y_test_pred = predict(X_val)
    acc_test = float(mean_squared_error(Y_val, Y_test_pred))
    Test_Accuracy[t]=1-acc_test

    t=t+1

  Results = pd.DataFrame()
  Results['Conv_curve']=Convergence_curve
  Results['Train_acc']=Train_Accuracy
  Results['Test_acc']=Test_Accuracy
  Results.to_csv('WOA_Conv_Train_Test_of_{}.csv'.format(dataset),header=True,index=False)

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Convergence_curve, label='Convergence Curve')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Fitness')
  plt.show()

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Train_Accuracy,label='Train Accuracy')
  plt.plot(Test_Accuracy, label='Test Accuracy')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  # performance evaluation of the selected feature subset
  features = np.concatenate(Sel_features, axis=0)
  asd = Counter(features)
  sorted_features = sorted(asd.items(), key=operator.itemgetter(1))
  high = asd.most_common(10)
  df = pd.DataFrame(high, columns=['Genes', 'Count'])
  df.to_csv('WOA_Selected_feature_of_{}.csv'.format(dataset),header=True,index=False)
  f1 = []
  for i in range(0,len(high)):
    f1.append(high[i][0])
  f1.append('class')
  data_new = data[f1]
  X_new = data_new.drop('class', axis=1)
  Y_new = data_new['class']
  smt = SMOTETomek(random_state=50,sampling_strategy='auto')
  X_bal_new, Y_bal_new = smt.fit_sample(X_new, Y_new)
  X_train_new, X_val_new, Y_train_new, Y_val_new = train_test_split(X_bal_new, Y_bal_new, test_size=0.3, stratify = Y_bal_new, random_state=None)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val = model.predict_proba(X_val_new)
  y_val1 = y_val[:,1]
  auc = roc_auc_score(Y_val_new, y_val1)
  fpr, tpr, thresholds = roc_curve(Y_val_new,y_val1)
  plot_roc_curve(fpr, tpr, auc, dataset)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val2 = model.predict(X_val_new)
  report = classification_report(Y_val_new,y_val2, output_dict=True)
  df1 = pd.DataFrame(report).transpose()
  df1.to_csv('WOA_Classification_report_of_selected_feature_subset_for_{}.csv'.format(dataset),header=True,index=True)

"""# Moth Flame Optimization"""

for dataset in ['OSCC','Breast','Colon','Leukemia','Ovarian','CNS']:
  X_train,X_val,Y_train,Y_val,X,data = initialization(dataset)
  SearchAgents_no = 50
  dim = X_train.shape[1]  
  Leader_pos = np.zeros(dim) 
  Leader_score = math.inf
  Max_iter = 100
  stepsize = np.zeros((SearchAgents_no,dim))
  Convergence_curve = np.zeros(Max_iter)
  Train_Accuracy = np.zeros(Max_iter)
  Test_Accuracy = np.zeros(Max_iter)
  Bias = 1
  ub = np.max(X_train,axis=0)
  lb = np.min(X_train,axis=0)
  Positions = np.zeros((SearchAgents_no,dim))
  Fitness = np.zeros(SearchAgents_no)
  i=0
  while i < SearchAgents_no:
    for j in range(0,dim):
      Positions[i][j] = ub[j]-(ub[j]-lb[j])*np.random.rand()
    i+=1
  Sel_features = []

  for i in range(0, SearchAgents_no):

    Y_pred = []
    
    for j in range(0, X_train.shape[0]):
      perceptron = np.dot(X_train[j],Positions[i]) + Bias
      sigmoid = 1.0/(1.0 + np.exp(-perceptron))
      Y_pred.append(sigmoid)
    
    fitness = log_loss(Y_train,Y_pred)
    Fitness[i]=fitness
    if fitness<Leader_score:
      Leader_score=fitness
      Leader_pos=Positions[i,:].copy() 

  for t in tqdm_notebook(range(Max_iter), total=Max_iter, unit="iterations"):

    flameNo = int(np.ceil(SearchAgents_no-(t+1)*((SearchAgents_no-1)/Max_iter)))

    Positions = np.clip(Positions, lb, ub)

    if t == 0:
      # Sort the first population of moths
      order = Fitness.argsort()
      Fitness = Fitness[order]
      Positions = Positions[order, :]

      # Update the flames
      bFlames = np.copy(Positions)
      bFlamesFit = np.copy(Fitness)

    else:
      # Sort the moths
      doublePop = np.vstack((bFlames, Positions))
      doubleFit = np.hstack((bFlamesFit, Fitness))

      order = doubleFit.argsort()
      doubleFit = doubleFit[order]
      doublePop = doublePop[order, :]

      # Update the flames
      bFlames = doublePop[:SearchAgents_no, :]
      bFlamesFit = doubleFit[:SearchAgents_no]

    # Update the position best flame obtained so far
    bFlameScore = bFlamesFit[0]
    bFlamesPos = bFlames[0, :]

    a=-1+(t+1)*((-1)/Max_iter)
    
    for i in range(0,SearchAgents_no):

      if i<=flameNo:

        distance_to_flame=abs(bFlames[i]-Positions[i])
        b=1
        z=(a-1)*np.random.rand(dim) + 1
        Positions[i]=distance_to_flame*np.exp(b*z)*np.cos(z*2*np.pi)+bFlames[i]
    
      if i>flameNo: 

        distance_to_flame=abs(bFlames[i]-Positions[i])
        b=1
        z=(a-1)*np.random.rand(dim) + 1
        Positions[i]=distance_to_flame*np.exp(b*z)*np.cos(z*2*np.pi)+bFlames[flameNo]
            
    for i in range(0, SearchAgents_no):
      Y_pred = []
      for j in range(0, X_train.shape[0]):
        perceptron = np.dot(X_train[j],Positions[i]) + Bias
        sigmoid = 1.0/(1.0 + np.exp(-perceptron))
        Y_pred.append(sigmoid)

      fitness = log_loss(Y_train,Y_pred)
      Fitness[i]=fitness
      # Update the leader
      if fitness<Leader_score: # Change this to > for maximization problem
        Leader_score=fitness # Update alpha
        Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position

    Convergence_curve[t]=Leader_score

    sel_features = imp_features(X,Positions)
    Sel_features.append(sel_features)

    Y_train_pred = predict(X_train)
    acc_train = float(mean_squared_error(Y_train, Y_train_pred))
    Train_Accuracy[t]=1-acc_train

    Y_test_pred = predict(X_val)
    acc_test = float(mean_squared_error(Y_val, Y_test_pred))
    Test_Accuracy[t]=1-acc_test

    t=t+1

  Results = pd.DataFrame()
  Results['Conv_curve']=Convergence_curve
  Results['Train_acc']=Train_Accuracy
  Results['Test_acc']=Test_Accuracy
  Results.to_csv('MFO_Conv_Train_Test_of_{}.csv'.format(dataset),header=True,index=False)

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Convergence_curve, label='Convergence Curve')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Fitness')
  plt.show()

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Train_Accuracy,label='Train Accuracy')
  plt.plot(Test_Accuracy, label='Test Accuracy')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  # performance evaluation of the selected feature subset
  features = np.concatenate(Sel_features, axis=0)
  asd = Counter(features)
  sorted_features = sorted(asd.items(), key=operator.itemgetter(1))
  high = asd.most_common(10)
  df = pd.DataFrame(high, columns=['Genes', 'Count'])
  df.to_csv('MFO_Selected_feature_of_{}.csv'.format(dataset),header=True,index=False)
  f1 = []
  for i in range(0,len(high)):
    f1.append(high[i][0])
  f1.append('class')
  data_new = data[f1]
  X_new = data_new.drop('class', axis=1)
  Y_new = data_new['class']
  smt = SMOTETomek(random_state=50,sampling_strategy='auto')
  X_bal_new, Y_bal_new = smt.fit_sample(X_new, Y_new)
  X_train_new, X_val_new, Y_train_new, Y_val_new = train_test_split(X_bal_new, Y_bal_new, test_size=0.3, stratify = Y_bal_new, random_state=None)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val = model.predict_proba(X_val_new)
  y_val1 = y_val[:,1]
  auc = roc_auc_score(Y_val_new, y_val1)
  fpr, tpr, thresholds = roc_curve(Y_val_new,y_val1)
  plot_roc_curve(fpr, tpr, auc, dataset)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val2 = model.predict(X_val_new)
  report = classification_report(Y_val_new,y_val2, output_dict=True)
  df1 = pd.DataFrame(report).transpose()
  df1.to_csv('MFO_Classification_report_of_selected_feature_subset_for_{}.csv'.format(dataset),header=True,index=True)

"""# Sine Cosine Algorithm"""

for dataset in ['OSCC','Breast','Colon','Leukemia','Ovarian','CNS']:
  X_train,X_val,Y_train,Y_val,X,data = initialization(dataset)
  SearchAgents_no = 50
  dim = X_train.shape[1]  
  Leader_pos = np.zeros(dim) 
  Leader_score = math.inf
  Max_iter = 100
  stepsize = np.zeros((SearchAgents_no,dim))
  Convergence_curve = np.zeros(Max_iter)
  Train_Accuracy = np.zeros(Max_iter)
  Test_Accuracy = np.zeros(Max_iter)
  Bias = 1
  ub = np.max(X_train,axis=0)
  lb = np.min(X_train,axis=0)
  Positions = np.zeros((SearchAgents_no,dim))
  Fitness = np.zeros(SearchAgents_no)
  i=0
  while i < SearchAgents_no:
    for j in range(0,dim):
      Positions[i][j] = ub[j]-(ub[j]-lb[j])*np.random.rand()
    i+=1
  Sel_features = []

  for i in range(0, SearchAgents_no):

    Y_pred = []
    
    for j in range(0, X_train.shape[0]):
      perceptron = np.dot(X_train[j],Positions[i]) + Bias
      sigmoid = 1.0/(1.0 + np.exp(-perceptron))
      Y_pred.append(sigmoid)
    
    fitness = log_loss(Y_train,Y_pred)
    Fitness[i]=fitness
    if fitness<Leader_score:
      Leader_score=fitness
      Leader_pos=Positions[i,:].copy() 

  for t in tqdm_notebook(range(Max_iter), total=Max_iter, unit="iterations"):

    a = 2

    r1=a-t*((a)/Max_iter)
    
    for i in range(0,SearchAgents_no):

      r2=(2*np.pi)*np.random.rand()
      r3=2*np.random.rand()
      r4=np.random.rand()
      
      if r4<0.5:
        Positions[i]= Positions[i]+(r1*np.sin(r2)*abs(r3*Leader_pos-Positions[i]))
      else:
        Positions[i]= Positions[i]+(r1*np.cos(r2)*abs(r3*Leader_pos-Positions[i]))
            
    for i in range(0, SearchAgents_no):
      Y_pred = []
      for j in range(0, X_train.shape[0]):
        perceptron = np.dot(X_train[j],Positions[i]) + Bias
        sigmoid = 1.0/(1.0 + np.exp(-perceptron))
        Y_pred.append(sigmoid)

      fitness = log_loss(Y_train,Y_pred)
      Fitness[i]=fitness
      # Update the leader
      if fitness<Leader_score: # Change this to > for maximization problem
        Leader_score=fitness # Update alpha
        Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position

    Convergence_curve[t]=Leader_score

    sel_features = imp_features(X,Positions)
    Sel_features.append(sel_features)

    Y_train_pred = predict(X_train)
    acc_train = float(mean_squared_error(Y_train, Y_train_pred))
    Train_Accuracy[t]=1-acc_train

    Y_test_pred = predict(X_val)
    acc_test = float(mean_squared_error(Y_val, Y_test_pred))
    Test_Accuracy[t]=1-acc_test

    t=t+1

  Results = pd.DataFrame()
  Results['Conv_curve']=Convergence_curve
  Results['Train_acc']=Train_Accuracy
  Results['Test_acc']=Test_Accuracy
  Results.to_csv('SCA_Conv_Train_Test_of_{}.csv'.format(dataset),header=True,index=False)

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Convergence_curve, label='Convergence Curve')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Fitness')
  plt.show()

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Train_Accuracy,label='Train Accuracy')
  plt.plot(Test_Accuracy, label='Test Accuracy')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  # performance evaluation of the selected feature subset
  features = np.concatenate(Sel_features, axis=0)
  asd = Counter(features)
  sorted_features = sorted(asd.items(), key=operator.itemgetter(1))
  high = asd.most_common(10)
  df = pd.DataFrame(high, columns=['Genes', 'Count'])
  df.to_csv('SCA_Selected_feature_of_{}.csv'.format(dataset),header=True,index=False)
  f1 = []
  for i in range(0,len(high)):
    f1.append(high[i][0])
  f1.append('class')
  data_new = data[f1]
  X_new = data_new.drop('class', axis=1)
  Y_new = data_new['class']
  smt = SMOTETomek(random_state=50,sampling_strategy='auto')
  X_bal_new, Y_bal_new = smt.fit_sample(X_new, Y_new)
  X_train_new, X_val_new, Y_train_new, Y_val_new = train_test_split(X_bal_new, Y_bal_new, test_size=0.3, stratify = Y_bal_new, random_state=None)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val = model.predict_proba(X_val_new)
  y_val1 = y_val[:,1]
  auc = roc_auc_score(Y_val_new, y_val1)
  fpr, tpr, thresholds = roc_curve(Y_val_new,y_val1)
  plot_roc_curve(fpr, tpr, auc, dataset)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val2 = model.predict(X_val_new)
  report = classification_report(Y_val_new,y_val2, output_dict=True)
  df1 = pd.DataFrame(report).transpose()
  df1.to_csv('SCA_Classification_report_of_selected_feature_subset_for_{}.csv'.format(dataset),header=True,index=True)

"""# Harris Hawk Optimization"""

for dataset in ['OSCC','Breast','Colon','Leukemia','Ovarian','CNS']:
  X_train,X_val,Y_train,Y_val,X,data = initialization(dataset)
  SearchAgents_no = 50
  dim = X_train.shape[1]  
  Leader_pos = np.zeros(dim) 
  Leader_score = math.inf
  Max_iter = 100
  stepsize = np.zeros((SearchAgents_no,dim))
  Convergence_curve = np.zeros(Max_iter)
  Train_Accuracy = np.zeros(Max_iter)
  Test_Accuracy = np.zeros(Max_iter)
  Bias = 1
  ub = np.max(X_train,axis=0)
  lb = np.min(X_train,axis=0)
  Positions = np.zeros((SearchAgents_no,dim))
  i=0
  while i < SearchAgents_no:
    for j in range(0,dim):
      Positions[i][j] = ub[j]-(ub[j]-lb[j])*np.random.rand()
    i+=1
  Sel_features = []

  for i in range(0, SearchAgents_no):

    Y_pred = []
    
    for j in range(0, X_train.shape[0]):
      perceptron = np.dot(X_train[j],Positions[i]) + Bias
      sigmoid = 1.0/(1.0 + np.exp(-perceptron))
      Y_pred.append(sigmoid)
    
    fitness = log_loss(Y_train,Y_pred)
    if fitness<Leader_score:
      Leader_score=fitness
      Leader_pos=Positions[i,:].copy() 

  for t in tqdm_notebook(range(Max_iter), total=Max_iter, unit="iterations"):

    for i in range(0, SearchAgents_no):
      E0 = 2 * np.random.uniform() - 1                        # -1 < E0 < 1
      E = 2 * E0 * (1 - (t + 1) * 1.0 / Max_iter)       # factor to show the decreasing energy of rabbit
      J = 2 * np.random.uniform() - 1 

      if (np.abs(E) >= 1):
        # Harris' hawks perch randomly based on 2 strategy:
        if (np.random.uniform() >= 0.5):        # perch based on other family members
          X_rand = deepcopy(Positions[np.random.randint(0, SearchAgents_no),:])
          Positions[i] = X_rand - np.random.uniform() * np.abs(X_rand - 2 * np.random.uniform() * Positions[i])

        else:           # perch on a random tall tree (random site inside group's home range)
          X_m = np.mean(Positions,axis=0)
          Positions[i] = (Leader_pos - X_m) - np.random.uniform()*(lb + np.random.uniform() * (ub-lb))

      else:
        
        if (np.random.uniform() >= 0.5):
          delta_X = Leader_pos - Positions[i]
          if (np.abs(E) >= 0.5):          # Hard besiege Eq. (6) in paper
            Positions[i] = delta_X - E * np.abs( J * Leader_pos - Positions[i])
          else:                           # Soft besiege Eq. (4) in paper
            Positions[i] = Leader_pos - E * np.abs(delta_X)

        else:
          xichma = np.power((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2.0)) / (gamma((1 + 1.5) * 1.5 * np.power(2, (1.5 - 1) / 2)) / 2.0), 1.0 / 1.5)
          LF_D = 0.01 * np.random.uniform() * xichma / np.power(np.abs(np.random.uniform()), 1.0 / 1.5)
          fit_F, fit_Z, fit_Y, Y, Z = None, None, None, np.zeros((dim)), np.zeros((dim))
          if (np.abs(E) >= 0.5):      # Soft besiege Eq. (10) in paper
            Y = Leader_pos - E * np.abs( J * Leader_pos - Positions[i] )
            Z = Y + np.random.uniform(-1, 1, dim) * LF_D

          else:                       # Hard besiege Eq. (11) in paper
            X_m = np.mean(Positions,axis=0)
            Y = Leader_pos - E * np.abs( J * Leader_pos - X_m )
            Z = Y + np.random.uniform(-1, 1, dim) * LF_D

          fit_Y = error(X_train,Y)

          fit_Z = error(X_train,Z)
              
          fit_F = error(X_train,Positions[i])

          if (fit_Y < fit_F):
            Positions[i] = Y
          if (fit_Z < fit_F):
            Positions[i] = Z
          
    for i in range(0, SearchAgents_no):
      Y_pred = []
      for j in range(0, X_train.shape[0]):
        perceptron = np.dot(X_train[j],Positions[i]) + Bias
        sigmoid = 1.0/(1.0 + np.exp(-perceptron))
        Y_pred.append(sigmoid)
      fitness = log_loss(Y_train,Y_pred)
      # Update the leader
      if fitness<Leader_score: # Change this to > for maximization problem
        Leader_score=fitness; # Update alpha
        Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position

    Convergence_curve[t]=Leader_score
    # Leader_pos = Positions[i,:].copy()

    sel_features = imp_features(X,Positions)
    Sel_features.append(sel_features)

    Y_train_pred = predict(X_train)
    acc_train = float(mean_squared_error(Y_train, Y_train_pred))
    Train_Accuracy[t]=1-acc_train
   

    Y_test_pred = predict(X_val)
    acc_test = float(mean_squared_error(Y_val, Y_test_pred))
    Test_Accuracy[t]=1-acc_test

    t=t+1

  Results = pd.DataFrame()
  Results['Conv_curve']=Convergence_curve
  Results['Train_acc']=Train_Accuracy
  Results['Test_acc']=Test_Accuracy
  Results.to_csv('HHO_Conv_Train_Test_of_{}.csv'.format(dataset),header=True,index=False)

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Convergence_curve, label='Convergence Curve')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Fitness')
  plt.show()

  plt.style.use("ggplot")
  plt.title(dataset)
  plt.plot(Train_Accuracy,label='Train Accuracy')
  plt.plot(Test_Accuracy, label='Test Accuracy')
  plt.legend(loc='best')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

  # performance evaluation of the selected feature subset
  features = np.concatenate(Sel_features, axis=0)
  asd = Counter(features)
  sorted_features = sorted(asd.items(), key=operator.itemgetter(1))
  high = asd.most_common(10)
  df = pd.DataFrame(high, columns=['Genes', 'Count'])
  df.to_csv('HHO_Selected_feature_of_{}.csv'.format(dataset),header=True,index=False)
  f1 = []
  for i in range(0,len(high)):
    f1.append(high[i][0])
  f1.append('class')
  data_new = data[f1]
  X_new = data_new.drop('class', axis=1)
  Y_new = data_new['class']
  smt = SMOTETomek(random_state=50,sampling_strategy='auto')
  X_bal_new, Y_bal_new = smt.fit_sample(X_new, Y_new)
  X_train_new, X_val_new, Y_train_new, Y_val_new = train_test_split(X_bal_new, Y_bal_new, test_size=0.3, stratify = Y_bal_new, random_state=None)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val = model.predict_proba(X_val_new)
  y_val1 = y_val[:,1]
  auc = roc_auc_score(Y_val_new, y_val1)
  fpr, tpr, thresholds = roc_curve(Y_val_new,y_val1)
  plot_roc_curve(fpr, tpr, auc, dataset)
  model = svm.SVC(gamma='auto', kernel='poly', degree=3, probability=True, max_iter=-1)
  model.fit(X_train_new,Y_train_new)
  y_val2 = model.predict(X_val_new)
  report = classification_report(Y_val_new,y_val2, output_dict=True)
  df1 = pd.DataFrame(report).transpose()
  df1.to_csv('HHO_Classification_report_of_selected_feature_subset_for_{}.csv'.format(dataset),header=True,index=True)