import numpy as np
from scipy.stats import bernoulli  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from random import seed
from random import gauss
from scipy.stats import truncnorm
from random import shuffle
import math
 

    

def NormVar(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def SeqChoose(v,K,t):
    """
    Returns which action to take: Pick Player1, Player 2, or run UCB

    Parameters
    ----------
    v : Rate
        Must be below 0.5
    K : Number of Arms
    t : Current time step

    Returns
    -------
    int
        Description of return value
        0,1,2 -> 0: UCB,  1: Player1, 2:Player 2
    """

    seq=0

    if t% (1/v)==0:
        seq = 0

    elif ((t%(1/v))+1) %  math.floor(1/(K*v)) == 0:

        seq = math.ceil((t%(1/v))/(math.floor(1/(K*v))))

    else:
        seq == 0
    

    return seq


#Create Data frame of Two Player Responses
p1=   NormVar(mean = .1, sd = .1, low = 0 , upp = 1  )
p2 =  NormVar(mean = .9, sd = .1, low = 0 , upp = 1  )
play1 = p1.rvs(10000)
play2 = p2.rvs(10000)
dat= {'p1': play1,'p2':play2}
df = pd.DataFrame(dat)

N = 100 # Number of times to iterate
d = 2  #Number of Arms
arms_selected = []  
numbers_of_selections = [1] * d  #Keep track of number of times each arm gets selected
sums_of_reward = [0] * d #Keep track of the total sum of rewards
total_reward = 0

#Initialization:pull each arm once
for i in range(0, d):

    sums_of_reward[i]  =  df.values[i, i]

for n in range(1, N):
    arm = -1
    v= 1/4 #rate 

    max_upper_bound = float('-inf')

    choose = SeqChoose(v,d,n) 

    if choose != 0: 
        choose = choose-1
        arms_selected.append(choose) # which arms have been selected
        numbers_of_selections[choose] += 1  # number of selections +1
        reward = df.values[n, choose] # get the reward of the arm
        sums_of_reward[choose] += reward # keep track of rewards of each arm
        total_reward +=reward




    else:
        #UCB 
        for i in range(0, d):

            #Confidence Interval
            conf_int = math.sqrt(2 * math.log(n) / numbers_of_selections[i])
            
            #Average Reward
            average_reward = sums_of_reward[i] / numbers_of_selections[i]
            
            #Upper Bound
            upper_bound = (average_reward + conf_int)

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                arm = i


        arms_selected.append(arm) # which arm have been selected
        numbers_of_selections[arm] += 1  # number of selections +1
        reward = df.values[n, arm] # get the reward of the arm 
        sums_of_reward[arm] += reward#keep track of rewards of each arm
        total_reward += reward



print(numbers_of_selections)
print(sums_of_reward)
print(total_reward)



