#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
from decimal import Decimal
import time
from copy import copy
import matplotlib.pyplot as plt


# In[2]:


def yukawas(Hb,vb,Hc,vc,Hd,HY,vY,phi1,theta1,phi2,theta2,Lambda_0):
        
    vd = np.zeros(3)
    
    def vol(nb,nc,nd,nY):
        v = nb*vb + nc*vc + nd*vd + nY*vY
        H = nb*Hb + nc*Hc + nd*Hd + nY*HY
        #print('v is {}'.format(v))
        #print('H is {}'.format(H))
        vol = np.dot(np.dot(v.T,np.linalg.inv(H)),v)
        #print('critical point is: {}'.format(vol))
        return vol
    
    #yukawa elements
    #def yukawa_element(P1,P2,P3):
    def ye(*Particles):
        p_sum = 0
        for p in Particles:
            p_sum += p
        #print('volume is: {}'.format(p_sum))
        return np.exp(-1./2. * np.abs(p_sum))
    
    
    #particle vols
    #print('calculating Q1')
    Q1 = vol(1,-1,-1,1)
    #print('Q1 vol is {}'.format(Q1))
    Q2 = vol(-1,-1,-1,1)
    Q3 = vol(0,-1,-1,1)
    
    uc1 = vol(1,-1,-1,-4)
    #print('calculating u2')
    uc2 = vol(-1,-1,-1,-4)
    uc3 = vol(0,-1,-1,-4)
    
    dc1 = vol(1,-1,3,2)
    dc2 = vol(-1,-1,3,2)
    dc3 = vol(0,-1,3,2)    
    
    L1 = vol(1,-1,3,-3)
    L2 = vol(-1,-1,3,-3)
    L3 = vol(0,-1,3,-3)
    
    Hu1 = vol(1,2,2,3)
    Hu2 = vol(-1,2,2,3)
    Hu3 = vol(0,2,2,3)
    
    Hd1 = vol(1,2,-2,-3)
    Hd2 = vol(-1,2,-2,-3)
    Hd3 = vol(0,2,-2,-3)
    
    ec1 = vol(1,-1,-1,6)
    ec2 = vol(-1,-1,-1,6)
    ec3 = vol(0,-1,-1,6)
    
    nu1 = vol(1,-1,5,0)
    nu2 = vol(-1,-1,5,0)
    nu3 = vol(0,-1,5,0)
    
    nuc1 = vol(1,-1,-5,0)
    nuc2 = vol(-1,-1,-5,0)
    nuc3 = vol(0,-1,-5,0)
    
    nuX = vol(0,-3,5,0)
    LX  = vol(0,-3,-3,3)
    
    Hu1mix = np.sin(theta1)*np.cos(phi1)
    Hu2mix = np.sin(theta1)*np.sin(phi1)
    Hu3mix = np.cos(theta1)
    
    Hd1mix = np.sin(theta2)*np.cos(phi2)
    Hd2mix = np.sin(theta2)*np.sin(phi2)
    Hd3mix = np.cos(theta2)
    
    yU12 = Hu3mix*ye(Q1,uc2,Hu3)
    #print('exp is {}'.format(ye(Q1,uc2,Hu3)))
    #print('yU12 is {}'.format(yU12))
    yU13 = Hu2mix*ye(Q1,uc3,Hu2)
    yU21 = Hu3mix*ye(Q2,uc1,Hu3)
    yU23 = Hu1mix*ye(Q2,uc3,Hu1)
    yU31 = Hu2mix*ye(Q3,uc1,Hu2)
    yU32 = Hu1mix*ye(Q3,uc2,Hu1)
    yU33 = Hu3mix*ye(Q3,uc3,Hu3)
    
    yD12 = Hd3mix*ye(Q1,dc2,Hd3)
    yD13 = Hd2mix*ye(Q1,dc3,Hd2)
    yD21 = Hd3mix*ye(Q2,dc1,Hd3)
    yD23 = Hd1mix*ye(Q2,dc3,Hd1)
    yD31 = Hd2mix*ye(Q3,dc1,Hd2)
    yD32 = Hd1mix*ye(Q3,dc2,Hd1)
    yD33 = Hd3mix*ye(Q3,dc3,Hd3)  
    
    yE12 = Hd3mix*ye(L1,ec2,Hd3)
    yE13 = Hd2mix*ye(L1,ec3,Hd2)
    yE21 = Hd3mix*ye(L2,ec1,Hd3)
    yE23 = Hd1mix*ye(L2,ec3,Hd1)
    yE31 = Hd2mix*ye(L3,ec1,Hd2)
    yE32 = Hd1mix*ye(L3,ec2,Hd1)
    yE33 = Hd3mix*ye(L3,ec3,Hd3)      
    
    #big matrix, {01}=nu1, {02}=nu2, {03}=nu3, {04}=nuX, {05}=nuc1, {06}=nuc2, {07}=nuc3
    # {10}=L1, {20}=L2, {30}=L3, {40}=LX, {50}=L1, {60}=L2, {70}=L3
    #YN12XX = ye(nuc1,nuc2,nuX,nuX)
    

    Yu = np.array([ [0,yU12,yU13], [yU21,0,yU23], [yU31,yU32,yU33] ])
    #print('Yu is :\n{}'.format(Yu))
    Yd = np.array([ [0,yD12,yD13], [yU21,0,yD23], [yD31,yD32,yD33] ])
    #print('Yd is :\n{}'.format(Yu))
    Ye = np.array([ [0,yE12,yE13], [yE21,0,yE23], [yE31,yE32,yE33] ])
    #print('Ye is :\n{}'.format(Yu))

    uU, sU, vhU = np.linalg.svd(Lambda_0*Yu)
    uD, sD, vhD = np.linalg.svd(Lambda_0*Yd)
    uE, sE, vhE = np.linalg.svd(Lambda_0*Ye)
    
    return sU,sD,sE

def yukawaData():
    YuMgut = 2.54*(10**-6)
    YcMgut = 1.37*(10**-3)
    YtMgut = 0.428
    
    dataU = np.array([YtMgut,YcMgut,YuMgut])
    
    YdMgut = 6.56*(10**-5)
    YsMgut = 1.24*(10**-4)
    YbMgut = 0.57*(10**-2)
    
    dataD = np.array([YbMgut,YsMgut,YdMgut])
    
    YeMgut  = 2.70341*(10**-6)
    YmuMgut = 5.70705*(10**-4)
    YtaMgut = 0.97020*(10**-2)

    dataE = np.array([YtaMgut,YmuMgut,YeMgut])
    
    return dataU, dataD, dataE

def yukawaError():
    
    eTop = 0.8
    eCharm = 2.7*(10.**-3.)
    eUp = 5.*(10.**-6.)
    
    errU = np.array([eTop,eCharm,eUp])
    
    eBottom = 1.*(10.**-1)
    eStrange = 2.5*(10.**-4)
    eDown = 1.2*(10.**-4)
    
    errD = np.array([eBottom,eStrange,eDown])
    
    eTau = 2.*(10.**-2)
    #eTau = 10.**10
    eMuon = 1.14*(10**-3)
    #eMuon = 10.**10
    eElectron = 5.04*(10.**-6)
    #eElectron = 10.**10
    #eTau = 5.
    #eMuon = 5.
    #eElectron = .

    errE = np.array([eTau,eMuon,eElectron])
    
    return errU, errD, errE


def yukawaErrorUpper():
    dataU,dataD,dataE = yukawaData()
    #eTop = 0.0017*dataU[0]
    eTop = 0.8
    #eCharm = 0.016*dataU[1]
    eCharm = 2.7*(10.**-3.)
    #eUp = 0.12*dataU[2]
    eUp = 5.*(10.**-6.)
    
    errU = np.array([eTop,eCharm,eUp])
    
    #eBottom = 0.0072*dataD[0]
    eBottom = 1.*(10.**-1)
    #eStrange = 0.12*data[1]
    eStrange = 2.5*(10.**-4)
    #eDown = 0.10*dataD[2]
    eDown = 1.2*(10.**-4)
    
    errD = np.array([eBottom,eStrange,eDown])
    
    #eTau = (6.8*10**-5)*dataE[0]
    eTau = 2.*(10.**-2)
    #eMuon = (2.3*(10**-8))*dataE[1]
    eMuon = 1.14*(10**-3)
    #eElectron = (6.1*10.**(-9))*dataE[2]
    eElectron = 10.**10
    errE = np.array([eTau,eMuon,eElectron])
    
    return errU, errD, errE
    

def yukawaErrorLower():
    dataU,dataD,dataE = yukawaData()
    eTop = 0.0017*dataU[0]
    eCharm = 0.016*dataU[1]
    eUp = 0.23*dataU[2]
    
    errU = np.array([eTop,eCharm,eUp])
    
    eBottom = 0.0048*dataD[0]
    eStrange = 0.054*dataD[1]
    eDown = 0.036*dataD[2]
    
    errD = np.array([eBottom,eStrange,eDown])
    
    #eTau = (6.8*10**-5)*dataE[0]
    eTau = (6.8*10**-4)*dataE[0]
    #eMuon = (2.3*(10**-8))*dataE[1]
    eMuon = (2.3*(10**-4))*dataE[1]
    #eElectron = (6.1*10.**(-9))*dataE[2]
    eElectron = (6.1*10.**(-4))*dataE[2]
    
    errE = np.array([eTau,eMuon,eElectron])
    
    return errU, errD, errE
    

def MSE(predict,true,error):
    errUpper = np.ndarray.flatten(np.array(yukawaErrorUpper()))
    errLower = np.ndarray.flatten(np.array(yukawaErrorLower()))
    loss = 0
    for i in range(9):
        diff = true[i]-predict[i]
        if diff < 0:
            loss += (diff/errUpper[i])**2
        else:
            loss += (diff/errLower[i])**2
    loss /= 9
        #loss = np.mean(((true-predict)/error)**2.)
    return loss

def params(cube):
    Hb = np.array([[cube[0],cube[1],cube[2]],[0.,cube[3],cube[4]],[0.,0.,0.]])
    Hc = np.array([[cube[5],cube[6],cube[7]],[0.,cube[8],cube[9]],[0.,0.,0.]])
    Hd = np.array([[cube[10],cube[11],cube[12]],[0.,cube[13],cube[14]],[0.,0.,0.]])
    HY = np.array([[cube[15],cube[16],cube[17]],[0.,cube[18],cube[19]],[0.,0.,0.]])
    #HY = np.zeros([3,3])
    
    #vb = np.array([cube[15],cube[16],cube[17]])
    vb = np.array([1,0,0])
    vc = np.array([cube[20],cube[21],cube[22]])
    vY = np.array([cube[23],cube[24],cube[25]])
    #vY = np.array([cube[16],cube[19],cube[20]])
    #vY = np.zeros(3)
                          
    phi1 = cube[26]
    theta1 = cube[27]
    phi2 = cube[28]
    theta2 = cube[29]
    L0 = cube[30]
    #phi1 = cube[21]
    #theta1 = cube[22]
    #phi2 = cube[23]
    #theta2 = cube[24]
    
    #Lambda_0 = ((np.pi * 4.)/24.)**(3./2.)
    
    Hb[2,2] = -Hb[0,0] - Hb[1,1]
    Hc[2,2] = -Hc[0,0] - Hc[1,1]
    Hd[2,2] = -Hd[0,0] - Hd[1,1]
    HY[2,2] = -HY[0,0] - HY[1,1]
    #for negative values, check det == 0
    for n in range(3):
        for m in range(n):
            Hb[n,m] = Hb[m,n]
            Hc[n,m] = Hc[m,n]
            Hd[n,m] = Hd[m,n]
            HY[n,m] = HY[m,n]
    
    return Hb,vb,Hc,vc,Hd,HY,vY,phi1,theta1,phi2,theta2,L0
    

#function for checking if params are valid
def params_check(cube):
    
    #cube is an array that holds all of the randomly generated numbers. They are assigned here.
    Hb = np.array([[cube[0],cube[1],cube[2]],[0.,cube[3],cube[4]],[0.,0.,0.]])
    Hc = np.array([[cube[5],cube[6],cube[7]],[0.,cube[8],cube[9]],[0.,0.,0.]])
    Hd = np.array([[cube[10],cube[11],cube[12]],[0.,cube[13],cube[14]],[0.,0.,0.]])
    HY = np.array([[cube[15],cube[16],cube[17]],[0.,cube[18],cube[19]],[0.,0.,0.]])

    Hb[2,2] = -Hb[0,0] - Hb[1,1]
    Hc[2,2] = -Hc[0,0] - Hc[1,1]
    Hd[2,2] = -Hd[0,0] - Hd[1,1]
    HY[2,2] = -HY[0,0] - HY[1,1]
    #for negative values, check det == 0
    for n in range(3):
        for m in range(n):
            Hb[n,m] = Hb[m,n]
            Hc[n,m] = Hc[m,n]
            Hd[n,m] = Hd[m,n]
            HY[n,m] = HY[m,n]
            
    def detf(H):
        det = -1*H[0,0]*(2*H[0,0]**2+2*H[0,1]**2+H[0,2]**2+H[1,2]**2)+(2*H[0,1]*H[0,2]*H[1,2])
        return det
            
        
    
    def det(nb,nc,nd,nY):
        H = nb*Hb + nc*Hc + nd*Hd + nY*HY
        det = detf(H)
        return det
    
    #particle dets
    Q1 = det(1,-1,-1,1)
    Q2 = det(-1,-1,-1,1)
    Q3 = det(0,-1,-1,1)
    
    uc1 = det(1,-1,-1,-4)
    uc2 = det(-1,-1,-1,-4)
    uc3 = det(0,-1,-1,-4)
    
    dc1 = det(1,-1,3,2)
    dc2 = det(-1,-1,3,2)
    dc3 = det(0,-1,3,2)    
    
    L1 = det(1,-1,3,-3)
    L2 = det(-1,-1,3,-3)
    L3 = det(0,-1,3,-3)
    
    Hu1 = det(1,2,2,3)
    Hu2 = det(-1,2,2,3)
    Hu3 = det(0,2,2,3)
    
    Hd1 = det(1,2,-2,-3)
    Hd2 = det(-1,2,-2,-3)
    Hd3 = det(0,2,-2,-3)
    
    ec1 = det(1,-1,-1,6)
    ec2 = det(-1,-1,-1,6)
    ec3 = det(0,-1,-1,6)
    
    nu1 = det(1,-1,5,0)
    nu2 = det(-1,-1,5,0)
    nu3 = det(0,-1,5,0)
    
    nuc1 = det(1,-1,-5,0)
    nuc2 = det(-1,-1,-5,0)
    nuc3 = det(0,-1,-5,0)
    
    nuX = det(0,-3,5,0)
    LX  = det(0,-3,-3,3)
    
    S1  = det(-1,4,0,0)
    S2  = det(1,4,0,0)
    S3  = det(0,4,0,0)

    #S1 & rh 
    #rhN
    #- all negative det
    # check superpotential
    
    # in both cases, the product of the determinants should be negative
    #if L1*L2*L3*nuc1*nuc2*nuX < 0:
    # check 
    lep_pos = L1 > 0 and L2 > 0 and L3 > 0
    lep_neg = L1 < 0 and L2 < 0 and L3 < 0

    print_dets = False
    final_print = True
    
    if (not lep_pos) and (not lep_neg): 
        if print_dets:
            print('Leptons Dets:\n')
            print(L1)
            print(L2)
            print(L3)

        return False

    # check neutrinos by counting number of positive and negative dets
    # Khoa: does order matter here, in (+,+,-) or (-,-,+) or can it be (+,-,+) etc.?

    if lep_neg:
        if nuc1 > 0 or nuc2 > 0 or nuc3 > 0 or nuX > 0:
            if print_dets:
                print('All Lepton Dets negative, Neutrino Dets:\n')
                print(nuc1)
                print(nuc2)
                print(nuc3)
                print(nuX)
            return False
    elif lep_pos:
        if nuc1 < 0 or nuc2 < 0 or nuc3 < 0 or nuX < 0:
            if print_dets:
                print('All Lepton Dets positive, Neutrino Dets:\n')
                print(nuc1)
                print(nuc2)
                print(nuc3)
                print(nuX)
                
            return False



    Ec = [ec1,ec2,ec3]
    Q = [Q1,Q2,Q3]
    Uc = [uc1,uc2,uc3]
    Dc = [dc1,dc2,dc3]
    S = [S1,S2,S3]
    Hu = [Hu1,Hu2,Hu3]
    Hd = [Hd1,Hd2,Hd3]
    nu = [nu1,nu2,nu3]

    ferm = [Ec,Q,Uc,Dc,S,Hu,Hd,nu]
    names = ['ec','Q','uc','dc','S','Hu','Hd','nu']
    j=0
    #set default to passing the condition, iterate over all fermions to check if any are wrong sign
    # wrong sign: any +'s if L->(-,-,-) and any -'s if L->(+,+,+)
    #pass_ferm = True
    if lep_pos:
        for fam in ferm:
            name = names[j]
            j+=1
            for part in fam:
                if part<0:
                    if print_dets:
                        print('All Lepton dets are positive, {} dets:\n'.format(name))
                        print(fam[0])
                        print(fam[1])
                        print(fam[2])
                    
                    return False
                
                
    elif lep_neg:
        for fam in ferm:
            name = names[j]
            j+=1
            for part in fam:
                if part>0: 
                    if print_dets:
                        print('All Lepton dets are negative, {} dets:\n'.format(name))
                        print(fam[0])
                        print(fam[1])
                        print(fam[2])
                    
                    return False

    
    L = [L1, L2, L3]
    nuc = [nuc1, nuc2, nuc3, nuX]
    
    ferm2 = [Ec,Q,Uc,Dc,S,Hu,Hd,L,nuc,nu]
    names2 = ['ec','Q','uc','dc','S','Hu','Hd','L','nuc','nu']
    i = 0
    k = 1
    if final_print:
        for fam in ferm2:
            k = 1
            for part in fam:
                print('The determinant of {}{} is {}\n'.format(names2[i],k,part))
                k += 1
            i += 1
    return True
    
    
    
def cols():
        columns=["y_top","y_charm","y_up","y_bottom","y_strange","y_down","y_tau","y_mu","y_e"            ,"Hb11","Hb12","Hb13","Hb21","Hb22","Hb23","Hb31","Hb32","Hb33"            ,"Hc11","Hc12","Hc13","Hc21","Hc22","Hc23","Hc31","Hc32","Hc33"            ,"Hd11","Hd12","Hd13","Hd21","Hd22","Hd23","Hd31","Hd32","Hd33"            ,"HY11","HY12","HY13","HY21","HY22","HY23","HY31","HY32","HY33"            ,"vb1","vb2","vb3","vc1","vc2","vc3"            ,"vY1","vY2","vY3","phi1","theta1","phi2","theta2","Lambda_0"]
        return columns

    
def cols_no_nan():
        columns=["y_top","y_charm","y_up","y_bottom","y_strange","y_down","y_tau","y_mu","y_e"             ,"Hb11","Hb12","Hb13","Hb22","Hb23"             ,"Hc11","Hc12","Hc13","Hc22","Hc23"             ,"Hd11","Hd12","Hd13","Hd22","Hd23"             ,"HY11","HY12","HY13","HY22","HY23"             ,"vb1","vb2","vb3","vc1","vc2","vc3"             ,"vY1","vY2","vY3","phi1","theta1","phi2","theta2","Lambda0"]
        return columns


# In[3]:


def grad(dx,cube,learning_rate):
    Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0 = params(cube)
    prediction = np.ndarray.flatten(np.array(yukawas(Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0)))
    data = np.ndarray.flatten(np.array(yukawaData())) 
    error = np.ndarray.flatten(np.array(yukawaError()))
    N = data.shape[0]
    dfdx = np.zeros(len(data))
    loss_old = MSE(prediction,data,error)
    for i in range(len(cube)):
        cube_prime = np.copy(cube)
        cube_prime[i] += dx

        Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0 = params(cube_prime)
        predict_prime = np.ndarray.flatten(np.array(yukawas(Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0)))
        dfdx = (predict_prime - prediction)/(dx)

        grad = 1./N*(np.dot(dfdx,((-2.*(-prediction+data)/(error**2.)))))
        
        
        if np.isnan(grad) or np.isinf(grad):
            grad = 0.

        while np.abs(grad) > np.abs(0.01*cube[i]): 
            grad *=0.01
            #print(cube[i])
            
        #grad = 1./N*(np.dot(dfdx,((-2.*(-prediction+data)))))

        cube[i] -= learning_rate * grad
        if not params_check(cube): cube[i] += learning_rate * grad
        Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0 = params(cube)
        prediction_new = np.ndarray.flatten(np.array(yukawas(Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0)))
        loss_new = MSE(prediction_new,data,error)
        #if(loss_new>loss_old): cube[i] += learning_rate * grad
        #else: loss_old = loss_new
        loss_old = loss_new
    Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0 = params(cube)
    prediction_final = np.ndarray.flatten(np.array(yukawas(Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0)))
    loss = MSE(prediction_final,data,error)#print loss
    return loss, cube


def optimize(cube,dx,learning_rate,N_iter,print_loss):
    min_cube = np.copy(cube)
    start_loss,start_cube = grad(dx,cube,learning_rate)
    min_loss = np.copy(start_loss)
    for n in range(N_iter):
        try:
            loss_new, cube_new = grad(dx,cube,learning_rate)
        except:
            pass
        
        print(loss_new)
        print(min_loss)
        
        
        if (loss_new < min_loss) and params_check(cube_new):
            min_loss = np.copy(loss_new)
            min_cube = np.copy(cube_new)
    
        if print_loss and ((n % 100) == 0):
            print('loss after iteration %i: %.2E' %(n, Decimal(loss_new)))
            print('Dloss after iteration %i: %f' %(n, loss_new))
            
    return min_cube, min_loss
            
def iterations(N_samples,N_iter,print_losses,write_data,write_file):

    columns=["y_top","y_charm","y_up","y_bottom","y_strange","y_down","y_tau","y_mu","y_e"                ,"Hb11","Hb12","Hb13","Hb22","Hb23"                ,"Hc11","Hc12","Hc13","Hc22","Hc23"                ,"Hd11","Hd12","Hd13","Hd22","Hd23"                ,"HY11","HY12","HY13","HY22","HY23"                ,"vc1","vc2","vc3"                ,"vY1","vY2","vY3"                ,"phi1","theta1","phi2","theta2","MLE"]
    n=0

    eigenlist = []
    
    while n < N_samples:
        eigenvals_cube = []
        if print_losses: print ("starting run " + str(int(n+1)))
        

        mycube = np.random.normal(1.,0.5,31) * (2*np.random.randint(0,2,size=(31))-1)
        
        mycube[26] = np.random.uniform(0,2*np.pi)
        mycube[27] = np.random.uniform(0,2*np.pi)
        mycube[28] = np.random.uniform(0,2*np.pi)
        mycube[29] = np.random.uniform(0,2*np.pi)
        a = time.time()
        my_params = params(mycube)
        b = time.time()
        print('Took {} seconds for {} calculations')
        
        
        #if params_check(mycube):
        dx = 0.001
        learning_rate = 0.01
        min_cube,min_loss = optimize(mycube,dx,learning_rate,N_iter,print_losses)
        if params_check(min_cube):
            Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0 = params(min_cube)
            sU,sD,sE = yukawas(Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0)
            for eigen in sU,sD,sE,min_cube:
                for val in eigen:
                    eigenvals_cube.append(val)
            eigenvals_cube.append(min_loss)
            eigenlist.append(eigenvals_cube)
            n+=1
            #else: continue
        else: None#print('check failed')
        
    
        
    df=pd.DataFrame(data=eigenlist,columns=columns)
        
    if(write_data):

        df.to_csv(write_file)
                
    #return sU, sD, sE, min_cube


# In[4]:


from tqdm import tnrange

def iter_local_min(N_samples,N_iter,cubeIndex1,cubeIndex2,dx,print_losses,write_data,write_file):

    columns=["y_top","y_charm","y_up","y_bottom","y_strange","y_down","y_tau","y_mu","y_e"                ,"Hb11","Hb12","Hb13","Hb22","Hb23"                ,"Hc11","Hc12","Hc13","Hc22","Hc23"                ,"Hd11","Hd12","Hd13","Hd22","Hd23"                ,"HY11","HY12","HY13","HY22","HY23"                ,"vc1","vc2","vc3"                ,"vY1","vY2","vY3"                ,"phi1","theta1","phi2","theta2","L0","MLE"]
    n=0

    eigenlist = []
            

    mycube = np.random.normal(1.,0.5,31) * (2*np.random.randint(0,2,size=(31))-1)

    mycube[26] = np.random.uniform(0,2*np.pi)
    mycube[27] = np.random.uniform(0,2*np.pi)
    mycube[28] = np.random.uniform(0,2*np.pi)
    mycube[29] = np.random.uniform(0,2*np.pi)
    
    mycube_2 = copy(mycube[cubeIndex2])
    print('starting cube1: {}\n'.format(cubeIndex1))
    for n in range(N_samples):
    #for n in tnrange(N_samples,desc='Cube 1:'):
        mycube[cubeIndex1] += dx
        mycube[cubeIndex2] = copy(mycube_2)
        print('starting cube2: {}, iter {}\n'.format(cubeIndex2,n))
        for m in range(N_samples):
        #for m in tnrange(N_samples,desc='Cube 2:'):
            
            eigenvals_cube = []
            mycube[cubeIndex2] += dx
            my_params = params(mycube)

            #if params_check(mycube):
            learning_rate = 0.01
            min_cube,min_loss = optimize(mycube,dx,learning_rate,N_iter,print_losses)
            #if params_check(min_cube):
            Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0 = params(min_cube)
            sU,sD,sE = yukawas(Hb,vb,Hc,vc,Hd,HY,vY,p1,t1,p2,t2,L0)
            for eigen in sU,sD,sE,min_cube:
                for val in eigen:
                    eigenvals_cube.append(val)
            eigenvals_cube.append(min_loss)
            eigenlist.append(eigenvals_cube)
            #n+=1
                #else: continue
            #else: None#print('check failed')
        
    
         
    df=pd.DataFrame(data=eigenlist,columns=columns)
        
    if(write_data):

        df.to_csv(write_file)
                

