#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


def unitvec(x_i, y_i):
    M_i = np.sqrt((np.diff(x_i)**2) + (np.diff(y_i)**2))
    V_i = np.array([np.diff(x_i)/M_i, np.diff(y_i)/M_i])
    return V_i


# In[3]:


def fit_vec(a, b):
    m, c = np.polyfit(a, b, 1) #fit line
    x = a
    y = x*m + c #get y from fit line    
    x = np.array([x.iloc[0], x.iloc[-1]]) #x-vec
    y = np.array([y.iloc[0], y.iloc[-1]]) #y-vec 
    return x,y


# In[4]:


def topview(x_i,y_i,x_f,y_f,x_mid,y_mid,x_stim,y_stim,tr,fig_file,zoom):
    #visualize stimulus and jump
    plt.plot(x_i,y_i,'-o', alpha = 0.3, color = 'k')
    plt.plot(x_f,y_f,'-o', alpha = 0.5, color = 'k')
    plt.plot(x_mid,y_mid, color = 'crimson', alpha = 0.8)
    plt.plot(x_stim,y_stim, color = 'steelblue', alpha = 0.8)

    plt.plot(tr['midpoint_x_top'][tr['jump']>0], 
             tr['midpoint_y_top'][tr['jump']>0], '.', alpha = 0.5, color = 'crimson')
    plt.plot(tr['stim_x'].dropna(), tr['stim_y'].dropna(), '.', alpha = 0.5, color = 'steelblue')

    plt.legend(['Initial Body Axis', 'Final Body Axis', 'Jump Trajectory', 'Stimulus Trajectory'])
    
    if (zoom == False): 
        plt.ylim([-720,0]); plt.xlim([0,1280])
    elif (zoom == True): 
        plt.ylim([-560,-160]); plt.xlim([440,840])
    plt.title('Top View'+'- $I_'+fig_file[-9]+'-Tr_'+fig_file[-1]+'$')
    plt.savefig(fig_file+"_topview",dpi = 300)
    plt.show();


# In[5]:


def sideview(tr,fig_file,zoom):
    #visualize side view
    plt.plot(tr['midpoint_x_side'][tr['jump']>0], tr['midpoint_y_side'][tr['jump']>0], '-o', color = 'indigo', alpha = 0.5)
    if (zoom == False): 
        plt.ylim([-720,0]); plt.xlim([0,1280])
    elif (zoom == True):
        plt.ylim([-670,-50]); plt.xlim([440,840])
    plt.legend(['Jump Trajectory'])
    plt.title('Side View'+'- $I_'+fig_file[-9]+'-Tr_'+fig_file[-1]+'$')
    plt.savefig(fig_file+"_sideview",dpi = 300)
    plt.show();


# In[6]:


def vec_view(V_i,V_f,V_mid,V_stim,fig_file):
    #visualize unit vectors
    plt.figure(figsize = (5,5))
    plt.quiver([0], [0], V_i[0], V_i[1], 
               angles='xy', scale_units='xy', scale=1, color = 'black', alpha = 0.3)
    
    plt.quiver([0], [0], V_f[0], V_f[1], 
               angles='xy', scale_units='xy', scale=1, color = 'black', alpha = 0.5)

    plt.quiver([0], [0], V_mid[0], V_mid[1], 
               angles='xy', scale_units='xy', scale=1, color = 'crimson')

    plt.quiver([0], [0], V_stim[0], V_stim[1], 
               angles='xy', scale_units='xy', scale=1, color = 'steelblue')

    plt.legend(['Initial Body Axis', 'Final Body Axis', 'Jump Trajectory', 'Stimulus Trajectory'], 
               loc = 'center right', borderaxespad = -13)
    plt.ylim([-1.2,1.2]); plt.xlim([-1.2,1.2]); plt.title('Unit Vectors'+'- $I_'+fig_file[-9]+'-Tr_'+fig_file[-1]+'$')
    plt.savefig(fig_file+"_vec_view",dpi = 300,bbox_inches='tight')
    plt.show();


# In[7]:


def preprocess(tr,fig_file,zoom=False,mea = 3):
    #mea is the number of frames after which we measure take off angle and velocity once the hopper 
    #has started jumping
    #this can be as low as 0 and as high as 10; every extra frame implies an additional 1/240th of a second
    
    #Segregate data into 'before jump' and 'after jump' using the *jump* vector
    tr['jump'] = np.cumsum(tr['jump'])

    #flip dltdv coordinates
    tr['front_y_top'] = -tr['front_y_top']
    tr['back_y_top'] = -tr['back_y_top']
    tr['stim_y'] = -tr['stim_y']
    tr['front_y_side'] = -tr['front_y_side']
    tr['back_y_side'] = -tr['back_y_side']
    
    #find midpoint of grasshopper in both views
    tr['midpoint_x_top'] = np.mean([tr['back_x_top'],tr['front_x_top']],axis=0)
    tr['midpoint_y_top'] = np.mean([tr['back_y_top'],tr['front_y_top']],axis=0)
    tr['midpoint_x_side'] = np.mean([tr['back_x_side'],tr['front_x_side']],axis=0)
    tr['midpoint_y_side'] = np.mean([tr['back_y_side'],tr['front_y_side']],axis=0)
    
    #fit a line to the stimulus midpoint and vectorize it
    x_stim, y_stim = fit_vec(tr['stim_x'].dropna(), tr['stim_y'].dropna())
    
    #fit a line to the grasshopper midpoint in top view and vectorize it
    x_mid, y_mid = fit_vec(tr['midpoint_x_top'][tr['jump']>0], tr['midpoint_y_top'][tr['jump']>0])
    
    #find body axis ten frames before grasshopper starts to jump and store it as a vector
    x_i = np.array([tr['back_x_top'][tr['jump']<1].iloc[-10],tr['front_x_top'][tr['jump']<1].iloc[-10]])
    y_i = np.array([tr['back_y_top'][tr['jump']<1].iloc[-10],tr['front_y_top'][tr['jump']<1].iloc[-10]])
    
    #find body axis one frame before grasshopper jumps and store it as a vector
    x_f = np.array([tr['back_x_top'][tr['jump']<1].iloc[-1],tr['front_x_top'][tr['jump']<1].iloc[-1]])
    y_f = np.array([tr['back_y_top'][tr['jump']<1].iloc[-1],tr['front_y_top'][tr['jump']<1].iloc[-1]])
    
    #construct unit vectors
    V_i = unitvec(x_i,y_i)
    V_f = unitvec(x_f,y_f)
    V_mid = unitvec(x_mid, y_mid)
    V_stim = unitvec(x_stim, y_stim)
    
    #derive parameters ##indicates parameter
    
    #from top view
    turn_ang = -np.arctan2(V_f[0]*V_i[1]-V_f[1]*V_i[0],V_f[0]*V_i[0]+V_f[1]*V_i[1])*180/np.pi ##
    azimuth = -np.arctan2(V_mid[0]*V_f[1]-V_mid[1]*V_f[0],V_mid[0]*V_f[0]+V_mid[1]*V_f[1])*180/np.pi ##
    neg_V_stim = -V_stim
    approach_angle = np.arctan2(neg_V_stim[0]*V_i[1]-neg_V_stim[1]*V_i[0],
                                neg_V_stim[0]*V_i[0]+neg_V_stim[1]*V_i[1])*180/np.pi ##
    
    #for azimuth and turning angle +ve angles imply movement to the left(clockwise) and -ve angles imply movement
    #to the right(counterclockwise)
    
    #for approach_angle, +ve implies hopper was facing to the right of the stimulus and -ve implies it
    #was facing left of the stimulus
    
    #from bottom view
    jump_top_x = tr['midpoint_x_top'][tr['jump']>0].iloc[mea] #we measure take-off angle and velocity 3 frames after hopper starts jumping
    jump_top_y = tr['midpoint_y_top'][tr['jump']>0].iloc[mea]

    jump_side_x = tr['midpoint_x_side'][tr['jump']>0].iloc[mea]
    jump_side_y = tr['midpoint_y_side'][tr['jump']>0].iloc[mea]

    pre_jump_top_x = tr['midpoint_x_top'][tr['jump']<1].iloc[-1]
    pre_jump_top_y = tr['midpoint_y_top'][tr['jump']<1].iloc[-1]

    pre_jump_side_x = tr['midpoint_x_side'][tr['jump']<1].iloc[-1]
    pre_jump_side_y = tr['midpoint_y_side'][tr['jump']<1].iloc[-1]

    jump_dist_top = np.sqrt((jump_top_x-pre_jump_top_x)**2 + (jump_top_y-pre_jump_top_y)**2)*0.037 #convert to cm
    jump_dist_side = np.sqrt((jump_side_x-pre_jump_side_x)**2 + (jump_side_y-pre_jump_side_y)**2)*0.018 #convert to cm

    jump_dist = np.sqrt(jump_dist_top**2 + jump_dist_side**2)
    #vector addition of distances

    height = np.abs(jump_side_y-pre_jump_side_y)*0.018 #convert to cm
    
    take_off_angle = np.arctan(height/jump_dist_top)*180/np.pi ##
    
    vel = jump_dist/(0.0125*mea) #divide by time #in cm/sec
    ##
    
    topview(x_i,y_i,x_f,y_f,x_mid,y_mid,x_stim,y_stim,tr,fig_file,zoom)
    sideview(tr,fig_file,zoom)
    vec_view(V_i,V_f,V_mid,V_stim,fig_file)
    
    return turn_ang, azimuth, approach_angle, take_off_angle, vel

def dist_plotter(a1,a2,a3,bin_size,Max,direction,fig_file,title,a,bt,mini,maxi):
    degrees = np.hstack([a1,a2,a3])
    a , b=np.histogram(degrees, bins=np.arange(a, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=bt, color='crimson', edgecolor='crimson', alpha = 0.5)
    ax.set_theta_zero_location(direction)
    ax.set_yticklabels([])
    ax.set_thetamin(mini)
    ax.set_thetamax(maxi)

    N = len(degrees)
    for i in range(0,N):
        if i<len(a1): ax.plot(a1[i]*np.pi/180,Max,'o',color = 'indigo',alpha = 0.5)
        if i<len(a2): ax.plot(a2[i]*np.pi/180,Max,'o',color = 'darkorange',alpha = 0.5)
        if i<len(a3): ax.plot(a3[i]*np.pi/180,Max,'o',color = 'teal',alpha = 0.5)
    fig.legend(['$I_1$', '$I_2$', '$I_3$'], loc = 'center left')
    plt.title(title, pad = 25)
    plt.savefig(fig_file+title,dpi = 300,bbox_inches='tight')
    plt.show()

