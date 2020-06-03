import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def circ_dist_plotter(a1,a2,a3,bin_size,radius,zero_direction,fig_file,title,start_ang,bottom,min_ax_ang,max_ax_ang):
    degrees = np.hstack([a1,a2,a3])
    a, b = np.histogram(degrees, bins=np.arange(start_ang, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=bottom, color='crimson', edgecolor='crimson', alpha = 0.5)
    ax.set_theta_zero_location(zero_direction)
    ax.set_yticklabels([])
    ax.set_thetamin(min_ax_ang)
    ax.set_thetamax(max_ax_ang)

    N = len(degrees)
    for i in range(0,N):
        if i<len(a1): ax.plot(a1[i]*np.pi/180,radius,'o',color = 'indigo',alpha = 0.5)
        if i<len(a2): ax.plot(a2[i]*np.pi/180,radius,'o',color = 'darkorange',alpha = 0.5)
        if i<len(a3): ax.plot(a3[i]*np.pi/180,radius,'o',color = 'teal',alpha = 0.5)
    fig.legend(['$I_1$', '$I_2$', '$I_3$'], loc = 'center left')
    plt.title(title, pad = 25)
    plt.savefig(fig_file+title,dpi = 300,bbox_inches='tight'); 
    plt.show()

    
def body_axis_plotter(X,Y,I,c,angle1,angle2,savename,zero_direction = "S"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = 'polar')
    for i in range(np.shape(X)[0]):
        x0 = X.iloc[i]*np.pi/180; x1 = Y.iloc[i]*np.pi/180
        plt.polar([x0,x1],[1,2],'-',color = c[int(I.iloc[i])-1],alpha = 0.8)
        plt.polar([x0,x0],[0,1],'--',color = c[int(I.iloc[i])-1],alpha = 0.8)
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.set_theta_zero_location(zero_direction)
    plt.ylim([0,2.1])
    fig.legend([angle2,angle1], loc = 'upper right')
    plt.savefig(savename, dpi = 300)
    plt.show()

    
def corr_plot(X,Y,ylab,xlab,savename):
    plt.figure(figsize = (5,5))
    plt.plot(X, Y, 'o')
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.savefig(savename, dpi = 300); plt.show()

    
def sign_plotter(X,Y,ylab,xlab,savename):
    sns.set_style('white')
    plt.figure(figsize = (5,5))
    plt.plot(np.sign(X)+0.1*np.random.rand(len(X)), 
             np.sign(Y)+0.1*np.random.rand(len(Y)), 'o',alpha = 0.2)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.savefig(savename, dpi = 300)
    plt.show(); sns.set()

    
def round_plotter(X,Y,I,c,title,savename,min_ax_ang = 0,max_ax_ang = 360,zero_direction = "S"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = 'polar')
    for i in range(np.shape(X)[0]):
        x0 = X.iloc[i]*np.pi/180; y0 = Y.iloc[i]
        plt.polar(x0,y0,'-o',color = c[int(I.iloc[i])-1],alpha = 0.8)
    plt.title(title, pad = 25)
    ax.set_thetamin(min_ax_ang)
    ax.set_thetamax(max_ax_ang)
    ax.set_theta_zero_location(zero_direction)
    plt.savefig(savename, dpi = 300)
    plt.show();

    
def spoke_plotter(X,Y,title,savename,min_ax_ang = 0,max_ax_ang = 360,zero_direction = "N"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = 'polar')
    for i in range(np.shape(X)[0]):
        x0 = X.iloc[i]*np.pi/180; x1 = Y.iloc[i]*np.pi/180
        plt.polar([x0,x1],[1,2],'-o',color = 'mediumvioletred',alpha = 0.8)
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    plt.title(title, pad = 25)
    plt.ylim([0,2.1])
    ax.set_thetamin(min_ax_ang)
    ax.set_thetamax(max_ax_ang)
    ax.set_theta_zero_location(zero_direction)
    plt.savefig(savename, dpi = 300)
    plt.show();
