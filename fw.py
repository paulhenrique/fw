# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:15:18 2016
@project: Simulações IMOC
@author: Ivan
"""

ver = 1.3

import json
from functools import partial
import numpy as np
import sys
import os
from scipy import constants
from scipy import special as scisp
from scipy.integrate import quad
from scipy.interpolate import spline
import matplotlib.pyplot as plt
import mpmath
from sympy import sympify
from timeit import default_timer as timer
from datetime import date
import time
import multiprocessing
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool

usuario_sistema = ""

def nope():
    stuff = ['I\'m sorry dave, I can\'t let you do that.',
             'Nope', 
             'Stop right there, criminal scum!', 
             'Are you crazy?', 
             'Why are you doing this?', 
             'Run as fast as you can and don\'t look back',
             'WRONG',
             'Please, stop this...']
    time.sleep(0.5)
    # print('!!!!Error!!!')
    time.sleep(0.5)
    # print(np.random.choice(stuff))
    time.sleep(0.7)
    
def krodel(i,j):
    if i == j:
        return 1
    elif i != j:
        return 0
    else:
        return nope()

#def menu():
    # print("Choose one option:")
    # print("Type '1' to test BesseJ for ordem and x*kphop(0)")
    # print("Type '2' to test Abs[Phi]² for ")
    # print("Type '3' to test gnmTME, gnmTMLA and gnmTMLAV graph")
    # print("Type '4' to test Cprz and x graphs in E, LA and LAV approachs")    
    # print("Type '5' to test gnmTME, gnmTMLA and gnmTMLAV numerically")
    # print("Type '6' to test tau and pi")
    # print("Type '7' to test an and bn")
    # print("Type '8' to make Fig.1")    
    # print("Type '9' to make Fig.2")   
    # print("Type '10' to make Fig.3")
    # print("Type '11' to make Fig.4")
    # print("Type '0' to exit.")
    
def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]

def gerar_arquivo_json(nome_arquivo, matriz, titulos):
     # with open(nome_arquivo, "w") as outfile: 
    # outfile = open("teste.json", "w")
    #var = json.dump([titulos, matriz], ensure_ascii=False, sort_keys=True, indent=1, separators=(',', ':'))
    data = json.dumps([titulos, matriz], sort_keys=True, indent=4, separators=(',', ':'))
    # print("<script>var nome_arquivo ='"+nome_arquivo+"'; </script>")
    global usuario_sistema
    fd = os.open("users/"+ usuario_sistema + "/" + nome_arquivo, os.O_RDWR|os.O_CREAT )
    arquivo = os.fdopen(fd, "w+")
    arquivo.write(data)
    arquivo.close()
    print(data)
    return 1234
    print(nome_arquivo)


'''|====|====|====|Parametros Gerais|====|====|====|'''

c = constants.c

pi = constants.pi

#permissividade do vácuo 
#eps0 = constants.epsilon_0

#permeabilidade do vácuo
#mu0 = constants.mu_0

#índice de refração do meio externo, água - parte real
indnr = 1.33

#índice de refração do meio externo - parte imaginária
indni = 0

#índice de refração do meio
indref = indnr - 1j*indni

#velocidade da luz no meio externo, água
vel = c/indnr

#comprimento de onda no vácuo
lambdzero = 1064*10**(-9)

#comprimento de onda no meio externo, água
lambd = lambdzero/indnr

#frequencia angular no meio externo, água
wo = 2*pi*(c/lambdzero)

#numero da onda no meio externo, água, para um unico feixe de Bessel de frequencia w
konda = (indnr*wo)/c

#raio da partícula,caso seja necessário usar mais adiante.
#raiop = 17.5*10**(-6)
raiop = lambd/20

#Índice de refração da esfera
npart = indnr*1.01
'''|====|====|====|====|====|====|====|'''

'''|====|====|====|Polarização|====|====|====|'''
'''pol = 1 para polarização em x; 
pol = 2 para polarização em y;
pol = 3 para polarização circular à direita;
pol = 4 para polarização circular à esquerda.
A POLARIZAÇÃO EM Y (pol = 2) SÓ FUNCIONA PARA FROZEN WAVES GERADAS COM BSCs A PARTIR DA ILA,
MÉTODO APROXIMADO. PARA O MÉTODO EXATO, HÁ APENAS A OPÇÃO DE POLARIZAÇÃO EM x OU CIRCULARES'''

pol = 1

'''|====|====|====|====|====|====|====|'''

'''|====|====|====|Parametros de Frozen Waves|====|====|====|'''

#Distancia máxima para o padrão de intensidade desejado da FW.
Zmax = None
L = None

#Numero de somatórias de ψ, ou seja, há 2NN+1 feixes de Bessel para compor a Frozen Wave
NN = None

#Parametro da FW
A = None

#Definição do padrão longitudinal desejado. Como está, define perfil constante
f = lambda _: None

#def f():
#    return 1

#Define qual Frozen Wave de artigos anteriores ou desejado pelo usuário será usada
tipo = None
Qbeta = None
spot = None
def checktipo():
    global NN, A, L, f, Zmax, Qbeta, spot
    if tipo == 0:
        NN = 0
        A = 0.99619475
        axicon0 = (11.48*pi)/180
    elif tipo == 1:  #FW1
        L = 1*10**(-3)
        NN = 15
        A = 0.95
        Zmax = 0.1
        f = lambda _: 1
    elif tipo == 2: 
        L = 2*10**(-3)
        NN = 15
        A = 0.98999999999
        f = lambda z: np.exp((3*z)/L)
        Zmax = 0.1
    elif tipo == 3:
        L = 1*10**(-3)
        NN = 15
        A = 0.9879999999999
        Zmax = 0.4
        dz = L/70
        l1 = (1.5*L/10) - dz
        l2 = (1.5*L/10) + dz
        l3 = -(1.5*L/10) - dz
        l4 = -(1.5*L/10) + dz
        f = lambda z: np.piecewise(z, 
            [(l1 <= z) and (z <= l2), 
             (l3 <= z) and (z <= l4), 
            z > l2, 
            z < l3, 
            l4 < z and z < l1],
            [lambda z: -4*((z-l1)*(z-l2)/np.square(l1-l2)), 
             lambda z: -4*np.sqrt(2)*((z-l3)*(z-l4)/np.square(l3-l4)), 
            lambda z: 0, 
            lambda z: 0, 
            lambda z: 0])
    elif tipo == 4: #FW2
        L = 1*10**(-3)
        NN = 15
        A = 0.9879999999999
        f = lambda _: 1
        Zmax = 0.1
    elif tipo == 5: #FW3
        L = 1*10**(-3)
        NN = 15
        A = 0.9879999999999
        f = lambda z: np.exp((5*z)/L)
        Zmax = 0.1
    elif tipo == 6:
        L = 7*10**(-4)
        NN = 15
        A = 0.98
        f = lambda z: np.piecewise(z, 
            [(-0.25*L <= z) and (z <= -0.24*L), 
             (-0.24*L <= z) and (z <= -0.1*L), 
            (-0.1*L <= z) and (z <= -0.09*L), 
            (-0.09*L <= z) and (z <= 0.09*L), 
            (0.09*L <= z) and (z <= 0.1*L),
            (0.1*L <= z) and (z <= 0.24*L),
            (0.24*L <= z) and (z <= 0.25*L),
            z > 0.25,
            z < -0.25],
            [lambda z: 1, 
             lambda z: 0, 
             lambda z: 1, 
             lambda z: 0, 
             lambda z: 1,
             lambda z: 0,
             lambda z: 1,
             lambda z: 0,
             lambda z: 0])
    elif tipo == 7:
        L = 1*10**(-3)
        NN = 25
        A = 0.97999999999
        f = lambda z: np.exp((5*z)/L)*mpmath.sin((3.25*z)/L)
        Zmax = 0.04
    elif tipo == 8:
        L = 1*10**(-3)
        NN = 15
        A = 0.9879999999999
        f = lambda z: np.exp(5*z/L)
        Zmax = 0.25 
    else:
        nope()
        raise SystemExit
    #Coeficiente Q
    Qbeta = (A*indnr*wo)/c
    #spot em relação a um feixe de bessel de frequencia w
    spot = 2.4/np.sqrt(np.square(konda)-np.square(Qbeta))

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Parametros Gerais parte 2|====|====|====|'''

def betaip(p):
    #parte imaginária da constante de fase, ou numero da onda longitudinal, dos 2NN+1 feixes de Bessel da superposição que compõe a Frozen Wave
    return (indni/indnr)*(Qbeta+(2*pi*p/L))
    '''(Observe que a oarte imaginária é nula, pois o meio externo suposto real não possui perdas)'''
    
def betarp(p):
    #parte real do numero da onda longitudinal para cada um dos 2NN+1 feixes de Bessel da superposição que compõe a Frozen Wave
    return Qbeta+(2*pi*p/L)

def betaii():
    #parte imaginaria do numero de onda longitudinal do feixe de Bessel central da supoerposição que compõe a Frozen Wave
    return (indni/indnr)*Qbeta
    '''Ou seja, betaii() = betaip(0)'''
    
def betap(p):
    #numero de onda longitudinal para qualquer um dos 2NN+1 feixes de Bessel da supoerposição. 'p' é o p-ésimo feixe de Bessel
    return betarp(p)-1j*betaip(p)

def kphop(p):
    #numero de onda radial (ou transversal) para cada feixe de Bessel da superposição que compõe a Frozen Wave
    return np.sqrt(np.square(konda)-np.square(betap(p)))
    
def kphopr(m):
    #parte real da componente transversal do vetor da onda, k_{phoq} com q = m
    return mpmath.re(kphop(m))
    
def kphopi(m):
    #parte imaginária da componente transversal do vetor da onda, k_{phoq} com q = m
    return mpmath.im(kphop(m))
    
def axicon(p):
    #ângulos de áxicon de cada feixe de Bessel que compõe a Frozen Wave
    #return mpmath.asin(mpmath.re(kphop(p))/konda)
    return np.arcsin(np.real(kphop(p))/konda)

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Localspot, spot e raio minimo|====|====|====|'''

#ordem inteira dos feixes de Bessel ou, consequentemente, da Frozen Wave
ordem = None
localspot = None
spot0 = None
R = None
valloc = None
def checkordem():
    global valloc, localspot, spot0, R
    if ordem == 0:
        valloc = 10**(-13)
    else:
        valloc = 2*10**(-6)

    LB = []
    for x in np.linspace(-valloc,valloc,num=1000):
        #LB.append([mpmath.besselj(ordem,(-1.18803*10**(-14))*mpmath.re(kphop(0))),x])
        LB.append([mpmath.besselj(ordem,x*mpmath.re(kphop(0))),x])
      
    localspot = max(LB)[1]
    
    #spot para o feixe de Bessel central de superposição que compõe a Frozen Wave no caso ordem = 0
    spot0 = float(2.405/mpmath.re(kphop(0)))

    #Raio mínimo da abertura física para geração eficiente da Frozen Wave
    R = 2*L*(mpmath.re(kphop(-NN))/mpmath.re(betap(-NN)))
'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Linha AP|====|====|====|'''

'''cx = "co"
while cx != "q":
   # #print("Choose one type of coeficient:")
   # #print("Type 'co' to calculate using a constant coeficient")
   # #print("Type 'z' to calculate using a 0")
   # #print("Type 'q' to quit")
   #cx = input("Option: ")
   if cx.lower() == "co":
        #AM constante
       def Ap(p):
           if tipo == 0:
               return krodel(p,0)
           else:
               fzt = lambda zt: f(zt)*np.exp(betaii()*zt)*np.exp(1j*((2 * pi * p)/L) * zt)
               return (1/L)*complex_quadrature(fzt,-Zmax*L,Zmax*L)
       break
   #elif cx.lower() == "z":
        #AM exponencial
       #def Ap(p):
           #fzt = lambda zt: 0
           #return (1/L)*complex_quadrature(fzt,-Zmax*L,Zmax*L)
       #break
   elif cx.lower() == "q":
       exit()
   else:
       nope()
       continue'''

def Ap(p):
    fzt = lambda zt: f(zt)*np.exp(betaii()*zt)*np.exp(1j*((2 * pi * p)/L) * zt)
    return (1/L)*complex_quadrature(fzt,-Zmax*L,Zmax*L)

def testeAp():
    #Am para intensidade constante 
    return list(map(Ap,range(-NN,NN+1)))
            

def Psi(rho, z, t, z0):
    soma = 0
    for p in range(-NN,NN+1):
        soma += testeAp()[p+NN]*mpmath.besselj(ordem, rho*kphop(p))*np.exp(-1j*(2*pi*p/(L))*(z-z0))*np.exp(-betaip(p)*(z-z0))
    return soma*np.exp(1j*wo*t)*np.exp(-1j*Qbeta*(z-z0))

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|BSCs - Fatores de forma - para frozen waves|====|====|====|'''

#Escolhe o tipo de aproximação localizada: 1 para original e 2 para modificada. Usaremos "1"
localiza = 1

#CC = 1 para Davies circularly symmetric BB; CC = 2 para ASR
CC = 1

def g(p):
    if CC==1:
        return (1+mpmath.cos(axicon(p)))/4
    else:
        return 0.5

#Primeira localização,referente ao método aproximado HILAL original
def Rloc(n):
    if localiza==1:
        return n + 0.5
    else:
        return np.sqrt((n-1)*(n+2))

#Segunda localização,referente ao método aproximado HILAL modificado
def Rloc2(n):
    return np.sqrt((n-1)*(n+2))

def tau(n,m,x):
    return -(1/(-1+(mpmath.cos(x))**2))*(
                    (-1-n)*mpmath.cos(x)*mpmath.legenp(n,m,mpmath.cos(x))+(1-m+n)*mpmath.legenp(1+n,m,mpmath.cos(x))
                    )*mpmath.sin(x)

def fpi(n,m,x):
    return mpmath.legenp(n,m,mpmath.cos(x))/mpmath.sin(x)
    
def Znm(n,m):
    if m==0:
        return 1j*((2*n*(n+1))/(2*n+1))
    else:
        return (-2*1j/(2*n+1))**(abs(m)-1)
    
'''|====|====|====|====|====|====|====|'''



'''|====|====|====|BSCs USING THE INTEGRAL LOCALIZED APPROXIMATION HILAL -
SÃO OS FATORES DE FORMA APROXIMADOS PARA FWs DE ORIGEM ESCALAR|====|====|====|'''

if pol==1:
    CoefA1 = 1
    CoefA2 = 1
    CoefB1 = 1j
    CoefB2 = 1j
elif pol==2:
    CoefA1 = 1j
    CoefA2 = -1j
    CoefB1 = -1
    CoefB2 = 1
elif pol==3:
    CoefA1 = 1
    CoefA2 = 1
    CoefB1 = 1j
    CoefB2 = 1j
else:
    CoefA1 = 1
    CoefA2 = 1
    CoefB1 = 1j
    CoefB2 = 1j
    
def gnmTMLAI(n,m,r0,phi0,z0):
    soma = 0
    for p in range(-NN,NN+1):
        soma += (1/2)*Znm(n,m)*testeAp()[p+NN]*(
            CoefA1*mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
            +CoefA2*mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(- 1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
            )*mpmath.exp(1j*betap(p)*z0)
    return soma
def gnmTELAI(n,m,r0,phi0,z0):
    soma = 0
    for p in range(-NN,NN+1):
        soma += (1/2)*Znm(n,m)*testeAp()[p+NN]*(
            CoefB1*mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
            -CoefB2*mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(- 1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
            )*mpmath.exp(1j*betap(p)*z0)
    return soma

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|BSCs USING THE ILA, BUT FOR CIRCULARLY SYMMETRIC FWs. 
THIS IS BETTER FOR COMPARISON WITH THE EXACT ONE, BUT ARE NOT THE ONE USED IN PREVIOUS WORKS.
INDEED, PREVIOUS WORKS RELIED ON THE BSCs SHOWN ABOVE|====|====|====|'''

def gnmTMLAVI(n,m,r0,phi0,z0):
    soma = 0
    for p in range(-NN,NN+1):
        soma += (1/2)*Znm(n,m)*(testeAp()[p+NN]/(g(p)*(1+mpmath.cos(axicon(p)))))*g(p)*((-1j)**ordem)*(
        (1+mpmath.cos(axicon(p)))*(mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
        +mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(-1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
        )+(1-mpmath.cos(axicon(p)))*(
        mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(-1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
        +mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
        ))*mpmath.exp(1j*betap(p)*z0)
    return soma
def gnmTELAVI(n,m,r0,phi0,z0):
    soma = 0
    for p in range(-NN,NN+1):
        soma += (1j/2)*Znm(n,m)*(testeAp()[p+NN]/(g(p)*(1+mpmath.cos(axicon(p)))))*g(p)*((-1j)**ordem)*(
        (1+mpmath.cos(axicon(p)))*(mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
        -mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(-1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
        )+(1-mpmath.cos(axicon(p)))*(
        -mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(-1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
        +mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
        ))*mpmath.exp(1j*betap(p)*z0)
    return soma

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|EXACT BSCs - VECTORIAL - JIAJIE. 
SÃO OS FATORES DE FORMA EXATOS APENAS PARA FEIXES CIRCULARMENTE SIMÉTRICOS, 
APENAS COM POLARIZAÇÃO ORIGINAL EM X|====|====|====|'''


def gnmTMEI(n,m,r0,phi0,z0):
    soma = 0
    for p in range(-NN,NN+1):
        soma += (testeAp()[p+NN]/(1+mpmath.cos(axicon(p))))*(
        (1j**(ordem-m+1))*np.exp(1j*(ordem - m + 1)*phi0)*mpmath.besselj(ordem-m+1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))+m*fpi(n,m,axicon(p)))
        +(1j**(ordem-m-1))*np.exp(1j*(ordem - m - 1)*phi0)*mpmath.besselj(ordem-m-1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))-m*fpi(n,m,axicon(p)))
        )*mpmath.exp(1j*betap(p)*z0)
    return (-(-1)**((m-abs(m))/2))*(mpmath.fac(n-m)/mpmath.fac(n+abs(m)))*soma        
def gnmTEEI(n,m,r0,phi0,z0):
    soma = 0
    for p in range(-NN,NN+1):
        soma += (testeAp()[p+NN]/(1+mpmath.cos(axicon(p))))*(
        (1j**(ordem-m+1))*np.exp(1j*(ordem - m + 1)*phi0)*mpmath.besselj(ordem-m+1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))+m*fpi(n,m,axicon(p)))
        -(1j**(ordem-m-1))*np.exp(1j*(ordem - m - 1)*phi0)*mpmath.besselj(ordem-m-1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))-m*fpi(n,m,axicon(p)))
        )*mpmath.exp(1j*betap(p)*z0)
    return (1j*(-1)**((m-abs(m))/2))*(mpmath.fac(n-m)/mpmath.fac(n+abs(m)))*soma  

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|BSCs setup|====|====|====|'''

if pol==1:
    def gnmTMLA(n,m,r0,phi0,z0):
        return gnmTMLAI(n,m,r0,phi0,z0)
    def gnmTELA(n,m,r0,phi0,z0):
        return gnmTELAI(n,m,r0,phi0,z0)
    def gnmTMLAV(n,m,r0,phi0,z0):
        return gnmTMLAVI(n,m,r0,phi0,z0)
    def gnmTELAV(n,m,r0,phi0,z0):
        return gnmTELAVI(n,m,r0,phi0,z0)
    def gnmTME(n,m,r0,phi0,z0):
        return gnmTMEI(n,m,r0,phi0,z0)
    def gnmTEE(n,m,r0,phi0,z0):
        return gnmTEEI(n,m,r0,phi0,z0)
elif pol==2:
    def gnmTMLA(n,m,r0,phi0,z0):
        return gnmTMLAI(n,m,r0,phi0,z0)
    def gnmTELA(n,m,r0,phi0,z0):
        return gnmTELAI(n,m,r0,phi0,z0)
    def gnmTMLAV(n,m,r0,phi0,z0):
        return gnmTMLAVI(n,m,r0,phi0,z0)
    def gnmTELAV(n,m,r0,phi0,z0):
        return -gnmTELAVI(n,m,r0,phi0,z0)
    def gnmTME(n,m,r0,phi0,z0):
        return gnmTEEI(n,m,r0,phi0,z0)
    def gnmTEE(n,m,r0,phi0,z0):
        return -gnmTMEI(n,m,r0,phi0,z0)
elif pol==3:
    def gnmTMLA(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTMLAI(n,m,r0,phi0,z0) - gnmTELAI(n,m,r0,phi0,z0))
    def gnmTELA(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTELAI(n,m,r0,phi0,z0) + gnmTMLAI(n,m,r0,phi0,z0))
    def gnmTMLAV(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTMLAVI(n,m,r0,phi0,z0) - gnmTELAVI(n,m,r0,phi0,z0))
    def gnmTELAV(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTELAVI(n,m,r0,phi0,z0) + gnmTELAVI(n,m,r0,phi0,z0))
    def gnmTME(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTEEI(n,m,r0,phi0,z0) - gnmTMEI(n,m,r0,phi0,z0))
    def gnmTEE(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTEEI(n,m,r0,phi0,z0) + gnmTMEI(n,m,r0,phi0,z0))
else:
    def gnmTMLA(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTMLAI(n,m,r0,phi0,z0) + gnmTELAI(n,m,r0,phi0,z0))
    def gnmTELA(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTELAI(n,m,r0,phi0,z0) - gnmTMLAI(n,m,r0,phi0,z0))
    def gnmTMLAV(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTMLAVI(n,m,r0,phi0,z0) + gnmTELAVI(n,m,r0,phi0,z0))
    def gnmTELAV(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTELAVI(n,m,r0,phi0,z0) - gnmTELAVI(n,m,r0,phi0,z0))
    def gnmTME(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTEEI(n,m,r0,phi0,z0) + gnmTMEI(n,m,r0,phi0,z0))
    def gnmTEE(n,m,r0,phi0,z0):
        return (1/np.sqrt(2))*(gnmTEEI(n,m,r0,phi0,z0) - gnmTMEI(n,m,r0,phi0,z0))
    
'''|====|====|====|====|====|====|====|'''



'''|====|====|====|COEFICIENTES DE MIE PARA PARTÍCULAS ESFÉRICAS|====|====|====|'''

#particula dielétrica, portanto sua permeabilidade é unitaria
mup = constants.mu_0 

def NEta(x,y):
    #Inverso da impedância intrínsica relativa da partícula
    return np.sqrt(abs(y)/(x*abs(mup/constants.mu_0)))

def mm(x,y):
    #índice de refração relativo da partícula, não entrará nos calculos neste momento
    return np.sign(y)*np.sqrt(np.abs(y/(x*(constants.mu_0/mup))))

def CMPsi(x,n):
    Psi = konda*x*scisp.sph_jn(n,konda*x)[0][n]
    dPsi = scisp.sph_jn(n,konda*x)[0][n]+ konda*x*(-scisp.sph_jn(n,konda*x)[0][n]/(2*konda*x) + ((1/2)*(scisp.sph_jn(-1+n,konda*x)[0][-1+n] - scisp.sph_jn(1+n,konda*x)[0][1+n])))
    return (Psi,dPsi)
    
def CMOme(x,n):
    Ome = konda*x*scisp.sph_yn(n,konda*x)[0][n]
    dOme = scisp.sph_yn(n,konda*x)[0][n]+ konda*x*(-scisp.sph_yn(n,konda*x)[0][n]/(2*konda*x) + ((1/2)*(scisp.sph_yn(-1+n,konda*x)[0][-1+n] - scisp.sph_yn(1+n,konda*x)[0][1+n])))
    return (Ome,dOme)

def CMRho(x,n):
    #Função de Ricatti-Bessel Gamman(x) e sua primeira derivada
    Psi, dPsi = CMPsi(x,n)
    Ome, dOme = CMOme(x,n)
    Rho = Psi + 1j*Ome
    dRho = dPsi + 1j*dOme
    return (Rho,dRho)
    
def CMKsi(x,n):
    #Função de Ricatti-Bessel ξn(x) e sua primeira derivada
    Psi, dPsi = CMPsi(x,n)
    Ome, dOme = CMOme(x,n)
    Ksi = Psi - 1j*Ome
    dKsi = dPsi - 1j*dOme
    return (Ksi,dKsi)

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Coeficientes de Mie para caso convencional - sem perdas|====|====|====|'''
    
def an(x,y,n):
    Psix, dPsix = CMPsi(x,n)
    Psisq, dPsisq = CMPsi(x*np.sqrt(y/x),n)
    Ksix, dKsix = CMKsi(x,n)
    return ((Psix*dPsisq)-(NEta(x,y)*Psisq*dPsix))/((Ksix*dPsisq)-(NEta(x,y)*Psisq*dKsix))

def bn(x,y,n):
    Psix, dPsix = CMPsi(x,n)
    Psisq, dPsisq = CMPsi(x*np.sqrt(y/x),n)
    Ksix, dKsix = CMKsi(x,n)
    return ((NEta(x,y)*Psix*dPsisq)-(Psisq*dPsix))/((NEta(x,y)*Ksix*dPsisq)-(Psisq*dKsix))

def cn(x,y,n):
    Psix, dPsix = CMPsi(x,n)
    Psisq, dPsisq = CMPsi(x*np.sqrt(y/x),n)
    Ksix, dKsix = CMKsi(x,n)
    return ((1j*mm(x,y))/((NEta(x,y)*Ksix*dPsisq)-(Psisq*dKsix)))

def dn(x,y,n):
    Psix, dPsix = CMPsi(x,n)
    Psisq, dPsisq = CMPsi(x*np.sqrt(y/x),n)
    Ksix, dKsix = CMKsi(x,n)
    return (1j*(x/y)/((NEta(x,y)*Ksix*dPsisq)-(Psisq*dKsix)))
    
'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Seção de choque longitudinal (Em Z) para partícula dielétrica - Exato e aproximado|====|====|====|'''

NMAX = 2
MMAX = 1
yy = 1.05**2

def CprzE(x,y,r0,phi0,z0):
    soma = 0
    for n in range(1,NMAX):
        for m in range(-min(n,MMAX),min(n,MMAX)):
            soma += (1/np.square(n+1))*(mpmath.fac(n+1+abs(m))/mpmath.fac(n - abs(m)))*mpmath.re(
                             (
                              (an(x,y,n) + np.conj(an(x,y,n+1)) - 2*an(x,y,n)*np.conj(an(x,y,n+1)))*gnmTME(n,m,r0,phi0,z0)*np.conj(gnmTME(n+1,m,r0,phi0,z0))
                             ) + (
                              (bn(x,y,n) + np.conj(bn(x,y,n+1))-2*bn(x,y,n)*np.conj(bn(x,y,n+1)))*gnmTEE(n,m,r0,phi0,z0)*np.conj(gnmTEE(n+1,m,r0,phi0,z0))
                             )
                             ) + m*((2*n+1)/(np.square(n)*np.square(n+1)))*(
                     mpmath.fac(n+abs(m))/mpmath.fac(n-abs(m)))*mpmath.re(1j*(
                     2*an(x,y,n)*np.conj(bn(x,y,n))-an(x,y,n)-np.conj(bn(x,y,n)))*gnmTME(n,m,r0,phi0,z0)*np.conj(gnmTEE(n,m,r0,phi0,z0)))
    return (np.square(lambd)/pi)*soma

def CprzLAV(x,y,r0,phi0,z0):
    soma = 0
    for n in range(1,NMAX):
        for m in range(-min(n,MMAX),min(n,MMAX)):
            soma += (1/np.square(n+1))*(mpmath.fac(n+1+abs(m))/mpmath.fac(n - abs(m)))*mpmath.re(
                             (
                              (an(x,y,n) + np.conj(an(x,y,n+1)) - 2*an(x,y,n)*np.conj(an(x,y,n+1)))*gnmTMLAV(n,m,r0,phi0,z0)*np.conj(gnmTMLAV(n+1,m,r0,phi0,z0))
                             ) + (
                              (bn(x,y,n) + np.conj(bn(x,y,n+1)) - 2*bn(x,y,n)*np.conj(bn(x,y,n+1)))*gnmTELAV(n,m,r0,phi0,z0)*np.conj(gnmTELAV(n+1,m,r0,phi0,z0))
                             )
                             ) + m*((2*n+1)/(np.square(n)*np.square(n+1)))*(
                     mpmath.fac(n+abs(m))/mpmath.fac(n - abs(m)))*mpmath.re(1j*(
                             2*an(x,y,n)*np.conj(bn(x,y,n))-an(x,y,n)-np.conj(bn(x,y,n)))*gnmTMLAV(n,m,r0,phi0,z0)*np.conj(gnmTELAV(n,m,r0,phi0,z0)))
    return (np.square(lambd)/pi)*soma
    
def CprzLA(x,y,r0,phi0,z0):
    soma = 0
    for n in range(1,NMAX):
        for m in range(-min(n,MMAX),min(n,MMAX)):
            soma += (1/np.square(n+1))*(mpmath.fac(n+1+abs(m))/mpmath.fac(n - abs(m)))*mpmath.re(
                             (
                              (an(x,y,n) + np.conj(an(x,y,n+1)) - 2*an(x,y,n)*np.conj(an(x,y,n+1)))*gnmTMLA(n,m,r0,phi0,z0)*np.conj(gnmTMLA(n+1,m,r0,phi0,z0))
                             ) + (
                              (bn(x,y,n) + np.conj(bn(x,y,n+1)) - 2*bn(x,y,n)*np.conj(bn(x,y,n+1)))*gnmTELA(n,m,r0,phi0,z0)*np.conj(gnmTELA(n+1,m,r0,phi0,z0))
                             )
                             ) + m*((2*n+1)/(np.square(n)*np.square(n+1)))*(
                     mpmath.fac(n+abs(m))/mpmath.fac(n - abs(m)))*mpmath.re(1j*(
                             2*an(x,y,n)*np.conj(bn(x,y,n))-an(x,y,n)-np.conj(bn(x,y,n)))*gnmTMLA(n,m,r0,phi0,z0)*np.conj(gnmTELA(n,m,r0,phi0,z0)))
    return (np.square(lambd)/pi)*soma
'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Seção de choque radial em X e Y para partícula dielétrica - Exato e aproximado|====|====|====|'''

def SnmpE(n,m,p,x,y,r0,phi0,z0):
    return (an(x,y,n)+np.conj(an(x,y,m)))*gnmTME(n,p,r0,phi0,z0)*np.conj(gnmTME(m,p+1,r0,phi0,z0))\
            + (bn(x,y,n)+np.conj(bn(x,y,m)))*gnmTEE(n,p,r0,phi0,z0)*np.conj(gnmTEE(m,p+1,r0,phi0,z0))
             
def SnmpLA(n,m,p,x,y,r0,phi0,z0):
    return (an(x,y,n)+np.conj(an(x,y,m)))*gnmTMLA(n,p,r0,phi0,z0)*np.conj(gnmTMLA(m,p+1,r0,phi0,z0))\
            + (bn(x,y,n)+np.conj(bn(x,y,m)))*gnmTELA(n,p,r0,phi0,z0)*np.conj(gnmTELA(m,p+1,r0,phi0,z0))
            
def SnmpLAV(n,m,p,x,y,r0,phi0,z0):
    return (an(x,y,n)+np.conj(an(x,y,m)))*gnmTMLAV(n,p,r0,phi0,z0)*np.conj(gnmTMLAV(m,p+1,r0,phi0,z0))\
            + (bn(x,y,n)+np.conj(bn(x,y,m)))*gnmTELAV(n,p,r0,phi0,z0)*np.conj(gnmTELAV(m,p+1,r0,phi0,z0))       
          
          
          
def TnmpE(n,m,p,x,y,r0,phi0,z0):
    return -1j*(an(x,y,n)+np.conj(bn(x,y,m)))*gnmTME(n,p,r0,phi0,z0)*np.conj(gnmTEE(m,p+1,r0,phi0,z0))\
            + 1j*(bn(x,y,n)+np.conj(an(x,y,m)))*gnmTEE(n,p,r0,phi0,z0)*np.conj(gnmTME(m,p+1,r0,phi0,z0))

def TnmpLA(n,m,p,x,y,r0,phi0,z0):
    return -1j*(an(x,y,n)+np.conj(bn(x,y,m)))*gnmTMLA(n,p,r0,phi0,z0)*np.conj(gnmTELA(m,p+1,r0,phi0,z0))\
            + 1j*(bn(x,y,n)+np.conj(an(x,y,m)))*gnmTELA(n,p,r0,phi0,z0)*np.conj(gnmTMLA(m,p+1,r0,phi0,z0))

def TnmpLAV(n,m,p,x,y,r0,phi0,z0):
    return -1j*(an(x,y,n)+np.conj(bn(x,y,m)))*gnmTMLAV(n,p,r0,phi0,z0)*np.conj(gnmTELAV(m,p+1,r0,phi0,z0))\
            + 1j*(bn(x,y,n)+np.conj(an(x,y,m)))*gnmTELAV(n,p,r0,phi0,z0)*np.conj(gnmTMLAV(m,p+1,r0,phi0,z0))



def UnmpE(n,m,p,x,y,r0,phi0,z0):      
    return (an(x,y,n)*np.conj(an(x,y,m)))*gnmTME(n,p,r0,phi0,z0)*np.conj(gnmTME(m,p+1,r0,phi0,z0))\
            + (bn(x,y,n)*np.conj(bn(x,y,m)))*gnmTEE(n,p,r0,phi0,z0)*np.conj(gnmTEE(m,p+1,r0,phi0,z0))

def UnmpLA(n,m,p,x,y,r0,phi0,z0):      
    return (an(x,y,n)*np.conj(an(x,y,m)))*gnmTMLA(n,p,r0,phi0,z0)*np.conj(gnmTMLA(m,p+1,r0,phi0,z0))\
            + (bn(x,y,n)*np.conj(bn(x,y,m)))*gnmTELA(n,p,r0,phi0,z0)*np.conj(gnmTELA(m,p+1,r0,phi0,z0))

def UnmpLAV(n,m,p,x,y,r0,phi0,z0):      
    return (an(x,y,n)*np.conj(an(x,y,m)))*gnmTMLAV(n,p,r0,phi0,z0)*np.conj(gnmTMLAV(m,p+1,r0,phi0,z0))\
            + (bn(x,y,n)*np.conj(bn(x,y,m)))*gnmTELAV(n,p,r0,phi0,z0)*np.conj(gnmTELAV(m,p+1,r0,phi0,z0))
            
            
            
def VnmpE(n,m,p,x,y,r0,phi0,z0):
    return 1j*(bn(x,y,n)*np.conj(an(x,y,m)))*gnmTEE(n,p,r0,phi0,z0)*np.conj(gnmTME(m,p+1,r0,phi0,z0))\
            - 1j*(an(x,y,n)*np.conj(bn(x,y,m)))*gnmTME(n,p,r0,phi0,z0)*np.conj(gnmTEE(m,p+1,r0,phi0,z0))
            
def VnmpLA(n,m,p,x,y,r0,phi0,z0):
    return 1j*(bn(x,y,n)*np.conj(an(x,y,m)))*gnmTELA(n,p,r0,phi0,z0)*np.conj(gnmTMLA(m,p+1,r0,phi0,z0))\
            - 1j*(an(x,y,n)*np.conj(bn(x,y,m)))*gnmTMLA(n,p,r0,phi0,z0)*np.conj(gnmTELA(m,p+1,r0,phi0,z0))

def VnmpLAV(n,m,p,x,y,r0,phi0,z0):
    return 1j*(bn(x,y,n)*np.conj(an(x,y,m)))*gnmTELAV(n,p,r0,phi0,z0)*np.conj(gnmTMLAV(m,p+1,r0,phi0,z0))\
            - 1j*(an(x,y,n)*np.conj(bn(x,y,m)))*gnmTMLAV(n,p,r0,phi0,z0)*np.conj(gnmTELAV(m,p+1,r0,phi0,z0))

def CprxE(x,y,r0,phi0,z0):
    soman=0
    somam=0
    for n in range(1,NMAX):
        for p in range(min(n,MMAX),1):
            somam += (mpmath.fac(n+p)/mpmath.fac(n-p))*(
                        mpmath.re(
                            SnmpE(n + 1, n, p - 1, x, y, r0, phi0, z0) + SnmpE(n, n + 1, -p, x, y, r0, phi0, z0) - 2*UnmpE(n + 1, n, p - 1, x, y, r0, phi0, z0) - 2*UnmpE(n, n + 1, -p, x, y, r0, phi0, z0)
                        ) * ((2*n+1)/(n**2)) * mpmath.re(
                            TnmpE(n, n, p - 1, x, y, r0, phi0, z0) - TnmpE(n, n, -p, x, y, r0, phi0, z0) - 2*VnmpE(n, n, p - 1, x, y, r0, phi0, z0) + 2*VnmpE(n, n, -p, x, y, r0, phi0, z0)
                        ) -(n+p+1)*(n+p+2)* mpmath.re(
                            SnmpE(n, n + 1, p, x, y, r0, phi0, z0) + SnmpE(n + 1, n, -p - 1, x, y, r0, phi0, z0) - 2*UnmpE(n, n + 1, p, x, y, r0, phi0, z0) - 2*UnmpE(n + 1, n, -p - 1, x, y, r0, phi0, z0)
                        )
                     )
        soman += (1/(n+1))*(
                    (n+2)*mpmath.re(
                    2*UnmpE(n,n+1,0,x,y,r0,phi0,z0) + 2*UnmpE(n+1, n, -1, x, y, r0, phi0, z0) - 2*SnmpE(n,n+1, 0, x, y, r0, phi0, z0) - 2*SnmpE(n + 1, n, -1, x, y, r0, phi0, z0)
                    ) + (1/(n+1))*somam)
    return ((lambd**2)/(2*pi))*soman
    
def CprxLAV(x,y,r0,phi0,z0):
    soman=0
    somam=0
    for n in range(1,NMAX):
        for p in range(min(n,MMAX),1):
            somam += (mpmath.fac(n+p)/mpmath.fac(n-p))*(
                        mpmath.re(
                            SnmpLAV(n + 1, n, p - 1, x, y, r0, phi0, z0) + SnmpLAV(n, n + 1, -p, x, y, r0, phi0, z0) - 2*UnmpLAV(n + 1, n, p - 1, x, y, r0, phi0, z0) - 2*UnmpLAV(n, n + 1, -p, x, y, r0, phi0, z0)
                        ) * ((2*n+1)/(n**2)) * mpmath.re(
                            TnmpLAV(n, n, p - 1, x, y, r0, phi0, z0) - TnmpLAV(n, n, -p, x, y, r0, phi0, z0) - 2*VnmpLAV(n, n, p - 1, x, y, r0, phi0, z0) + 2*VnmpLAV(n, n, -p, x, y, r0, phi0, z0)
                        ) -(n+p+1)*(n+p+2)* mpmath.re(
                            SnmpLAV(n, n + 1, p, x, y, r0, phi0, z0) + SnmpLAV(n + 1, n, -p - 1, x, y, r0, phi0, z0) - 2*UnmpLAV(n, n + 1, p, x, y, r0, phi0, z0) - 2*UnmpLAV(n + 1, n, -p - 1, x, y, r0, phi0, z0)
                        )
                     )
        soman += (1/(n+1))*(
                    (n+2)*mpmath.re(
                    2*UnmpLAV(n,n+1,0,x,y,r0,phi0,z0) + 2*UnmpLAV(n+1, n, -1, x, y, r0, phi0, z0) - 2*SnmpLAV(n,n+1, 0, x, y, r0, phi0, z0) - 2*SnmpLAV(n + 1, n, -1, x, y, r0, phi0, z0)
                    ) + (1/(n+1))*somam)
    return ((lambd**2)/(2*pi))*soman

def CprxLA(x,y,r0,phi0,z0):
    soman=0
    somam=0
    for n in range(1,NMAX):
        for p in range(min(n,MMAX),1):
            somam += (mpmath.fac(n+p)/mpmath.fac(n-p))*(
                        mpmath.re(
                            SnmpLA(n + 1, n, p - 1, x, y, r0, phi0, z0) + SnmpLA(n, n + 1, -p, x, y, r0, phi0, z0) - 2*UnmpLA(n + 1, n, p - 1, x, y, r0, phi0, z0) - 2*UnmpLA(n, n + 1, -p, x, y, r0, phi0, z0)
                        ) * ((2*n+1)/(n**2)) * mpmath.re(
                            TnmpLA(n, n, p - 1, x, y, r0, phi0, z0) - TnmpLA(n, n, -p, x, y, r0, phi0, z0) - 2*VnmpLA(n, n, p - 1, x, y, r0, phi0, z0) + 2*VnmpLA(n, n, -p, x, y, r0, phi0, z0)
                        ) -(n+p+1)*(n+p+2)* mpmath.re(
                            SnmpLA(n, n + 1, p, x, y, r0, phi0, z0) + SnmpLA(n + 1, n, -p - 1, x, y, r0, phi0, z0) - 2*UnmpLA(n, n + 1, p, x, y, r0, phi0, z0) - 2*UnmpLA(n + 1, n, -p - 1, x, y, r0, phi0, z0)
                        )
                     )
        soman += (1/(n+1))*(
                    (n+2)*mpmath.re(
                    2*UnmpLA(n,n+1,0,x,y,r0,phi0,z0) + 2*UnmpLA(n+1, n, -1, x, y, r0, phi0, z0) - 2*SnmpLA(n,n+1, 0, x, y, r0, phi0, z0) - 2*SnmpLA(n + 1, n, -1, x, y, r0, phi0, z0)
                    ) + (1/(n+1))*somam)
    return ((lambd**2)/(2*pi))*soman

def CpryE(x,y,r0,phi0,z0):
    soman=0
    somam=0
    for n in range(1,NMAX):
        for p in range(min(n,MMAX),1):
            somam += (mpmath.fac(n+p)/mpmath.fac(n-p))*(
                        mpmath.im(
                            SnmpE(n + 1, n, p - 1, x, y, r0, phi0, z0) + SnmpE(n, n + 1, -p, x, y, r0, phi0, z0) - 2*UnmpE(n + 1, n, p - 1, x, y, r0, phi0, z0) - 2*UnmpE(n, n + 1, -p, x, y, r0, phi0, z0)
                        ) * ((2*n+1)/(n**2)) * mpmath.im(
                            TnmpE(n, n, p - 1, x, y, r0, phi0, z0) - TnmpE(n, n, -p, x, y, r0, phi0, z0) - 2*VnmpE(n, n, p - 1, x, y, r0, phi0, z0) + 2*VnmpE(n, n, -p, x, y, r0, phi0, z0)
                        ) -(n+p+1)*(n+p+2)* mpmath.im(
                            SnmpE(n, n + 1, p, x, y, r0, phi0, z0) + SnmpE(n + 1, n, -p - 1, x, y, r0, phi0, z0) - 2*UnmpE(n, n + 1, p, x, y, r0, phi0, z0) - 2*UnmpE(n + 1, n, -p - 1, x, y, r0, phi0, z0)
                        )
                     )
        soman += (1/(n+1))*(
                    (n+2)*mpmath.im(
                    2*UnmpE(n,n+1,0,x,y,r0,phi0,z0) + 2*UnmpE(n+1, n, -1, x, y, r0, phi0, z0) - 2*SnmpE(n,n+1, 0, x, y, r0, phi0, z0) - 2*SnmpE(n + 1, n, -1, x, y, r0, phi0, z0)
                    ) + (1/(n+1))*somam)
    return ((lambd**2)/(2*pi))*soman

def CpryLAV(x,y,r0,phi0,z0):
    soman=0
    somam=0
    for n in range(1,NMAX):
        for p in range(min(n,MMAX),1):
            somam += (mpmath.fac(n+p)/mpmath.fac(n-p))*(
                        mpmath.im(
                            SnmpLAV(n + 1, n, p - 1, x, y, r0, phi0, z0) + SnmpLAV(n, n + 1, -p, x, y, r0, phi0, z0) - 2*UnmpLAV(n + 1, n, p - 1, x, y, r0, phi0, z0) - 2*UnmpLAV(n, n + 1, -p, x, y, r0, phi0, z0)
                        ) * ((2*n+1)/(n**2)) * mpmath.im(
                            TnmpLAV(n, n, p - 1, x, y, r0, phi0, z0) - TnmpLAV(n, n, -p, x, y, r0, phi0, z0) - 2*VnmpLAV(n, n, p - 1, x, y, r0, phi0, z0) + 2*VnmpLAV(n, n, -p, x, y, r0, phi0, z0)
                        ) -(n+p+1)*(n+p+2)* mpmath.im(
                            SnmpLAV(n, n + 1, p, x, y, r0, phi0, z0) + SnmpLAV(n + 1, n, -p - 1, x, y, r0, phi0, z0) - 2*UnmpLAV(n, n + 1, p, x, y, r0, phi0, z0) - 2*UnmpLAV(n + 1, n, -p - 1, x, y, r0, phi0, z0)
                        )
                     )
        soman += (1/(n+1))*(
                    (n+2)*mpmath.im(
                    2*UnmpLAV(n,n+1,0,x,y,r0,phi0,z0) + 2*UnmpLAV(n+1, n, -1, x, y, r0, phi0, z0) - 2*SnmpLAV(n,n+1, 0, x, y, r0, phi0, z0) - 2*SnmpLAV(n + 1, n, -1, x, y, r0, phi0, z0)
                    ) + (1/(n+1))*somam)
    return ((lambd**2)/(2*pi))*soman

def CpryLA(x,y,r0,phi0,z0):
    soman=0
    somam=0
    for n in range(1,NMAX):
        for p in range(min(n,MMAX),1):
            somam += (mpmath.fac(n+p)/mpmath.fac(n-p))*(
                        mpmath.im(
                            SnmpLA(n + 1, n, p - 1, x, y, r0, phi0, z0) + SnmpLA(n, n + 1, -p, x, y, r0, phi0, z0) - 2*UnmpLA(n + 1, n, p - 1, x, y, r0, phi0, z0) - 2*UnmpLA(n, n + 1, -p, x, y, r0, phi0, z0)
                        ) * ((2*n+1)/(n**2)) * mpmath.im(
                            TnmpLA(n, n, p - 1, x, y, r0, phi0, z0) - TnmpLA(n, n, -p, x, y, r0, phi0, z0) - 2*VnmpLA(n, n, p - 1, x, y, r0, phi0, z0) + 2*VnmpLA(n, n, -p, x, y, r0, phi0, z0)
                        ) -(n+p+1)*(n+p+2)* mpmath.im(
                            SnmpLA(n, n + 1, p, x, y, r0, phi0, z0) + SnmpLA(n + 1, n, -p - 1, x, y, r0, phi0, z0) - 2*UnmpLA(n, n + 1, p, x, y, r0, phi0, z0) - 2*UnmpLA(n + 1, n, -p - 1, x, y, r0, phi0, z0)
                        )
                     )
        soman += (1/(n+1))*(
                    (n+2)*mpmath.im(
                    2*UnmpLA(n,n+1,0,x,y,r0,phi0,z0) + 2*UnmpLA(n+1, n, -1, x, y, r0, phi0, z0) - 2*SnmpLA(n,n+1, 0, x, y, r0, phi0, z0) - 2*SnmpLA(n + 1, n, -1, x, y, r0, phi0, z0)
                    ) + (1/(n+1))*somam)
    return ((lambd**2)/(2*pi))*soman

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Torques ópticos - EXATO E APROXIMADO|====|====|====|'''

def TxLA(x,y,r0,phi0,z0):
    soma = 0
    for p in range(1,MMAX):
        for n in range(p,NMAX):
            soma += ((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*mpmath.re(
                        (
                            gnmTMLA(n,p-1,r0,phi0,z0)*np.conj(gnmTMLA(n,p,r0,phi0,z0)) - gnmTMLA(n,-p,r0,phi0,z0)*np.conj(gnmTMLA(n,-p+1,r0,phi0,z0))
                        )*(
                            2*np.square(abs(an(x,y,n)))-(an(x,y,n)+np.conj(an(x,y,n)))
                        )+(
                            gnmTELA(n,p-1,r0,phi0,z0)*np.conj(gnmTELA(n,p,r0,phi0,z0)) - gnmTELA(n,-p,r0,phi0,z0)*np.conj(gnmTELA(n,-p+1,r0,phi0,z0))                        
                        )*(
                            2*np.square(abs(bn(x,y,n)))-(bn(x,y,n)+np.conj(bn(x,y,n)))
                        )
                    )
    return ((-2*np.sqrt(y/x))/c)*(pi/(konda**3))*soma
    
def TxLAV(x,y,r0,phi0,z0):
    soma = 0
    for p in range(1,MMAX):
        for n in range(p,NMAX):
            soma += ((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*mpmath.re(
                        (
                            gnmTMLAV(n,p-1,r0,phi0,z0)*np.conj(gnmTMLAV(n,p,r0,phi0,z0)) - gnmTMLAV(n,-p,r0,phi0,z0)*np.conj(gnmTMLAV(n,-p+1,r0,phi0,z0))
                        )*(
                            2*np.square(abs(an(x,y,n)))-(an(x,y,n)+np.conj(an(x,y,n)))
                        )+(
                            gnmTELAV(n,p-1,r0,phi0,z0)*np.conj(gnmTELAV(n,p,r0,phi0,z0)) - gnmTELAV(n,-p,r0,phi0,z0)*np.conj(gnmTELAV(n,-p+1,r0,phi0,z0))                        
                        )*(
                            2*np.square(abs(bn(x,y,n)))-(bn(x,y,n)+np.conj(bn(x,y,n)))
                        )
                    )
    return ((-2*np.sqrt(y/x))/c)*(pi/(konda**3))*soma

def TxE(x,y,r0,phi0,z0):
    soma = 0
    for p in range(1,MMAX):
        for n in range(p,NMAX):
            soma += ((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*mpmath.re(
                        (
                            gnmTME(n,p-1,r0,phi0,z0)*np.conj(gnmTME(n,p,r0,phi0,z0)) - gnmTME(n,-p,r0,phi0,z0)*np.conj(gnmTME(n,-p+1,r0,phi0,z0))
                        )*(
                            2*np.square(abs(an(x,y,n)))-(an(x,y,n)+np.conj(an(x,y,n)))
                        )+(
                            gnmTEE(n,p-1,r0,phi0,z0)*np.conj(gnmTEE(n,p,r0,phi0,z0)) - gnmTEE(n,-p,r0,phi0,z0)*np.conj(gnmTEE(n,-p+1,r0,phi0,z0))                        
                        )*(
                            2*np.square(abs(bn(x,y,n)))-(bn(x,y,n)+np.conj(bn(x,y,n)))
                        )
                    )
    return ((-2*np.sqrt(y/x))/c)*(pi/(konda**3))*soma
    
def TyLA(x,y,r0,phi0,z0):
    soma = 0
    for p in range(1,MMAX):
        for n in range(p,NMAX):
            soma += ((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*mpmath.im(
                        (
                            gnmTMLA(n,p-1,r0,phi0,z0)*np.conj(gnmTMLA(n,p,r0,phi0,z0)) - gnmTMLA(n,-p,r0,phi0,z0)*np.conj(gnmTMLA(n,-p+1,r0,phi0,z0))
                        )*(
                            2*np.square(abs(an(x,y,n)))-(an(x,y,n)+np.conj(an(x,y,n)))
                        )+(
                            gnmTELA(n,p-1,r0,phi0,z0)*np.conj(gnmTELA(n,p,r0,phi0,z0)) - gnmTELA(n,-p,r0,phi0,z0)*np.conj(gnmTELA(n,-p+1,r0,phi0,z0))                        
                        )*(
                            2*np.square(abs(bn(x,y,n)))-(bn(x,y,n)+np.conj(bn(x,y,n)))
                        )
                    )
    return ((-2*np.sqrt(y/x))/c)*(pi/(konda**3))*soma
    
def TyLAV(x,y,r0,phi0,z0):
    soma = 0
    for p in range(1,MMAX):
        for n in range(p,NMAX):
            soma += ((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*mpmath.im(
                        (
                            gnmTMLAV(n,p-1,r0,phi0,z0)*np.conj(gnmTMLAV(n,p,r0,phi0,z0)) - gnmTMLAV(n,-p,r0,phi0,z0)*np.conj(gnmTMLAV(n,-p+1,r0,phi0,z0))
                        )*(
                            2*np.square(abs(an(x,y,n)))-(an(x,y,n)+np.conj(an(x,y,n)))
                        )+(
                            gnmTELAV(n,p-1,r0,phi0,z0)*np.conj(gnmTELAV(n,p,r0,phi0,z0)) - gnmTELAV(n,-p,r0,phi0,z0)*np.conj(gnmTELAV(n,-p+1,r0,phi0,z0))                        
                        )*(
                            2*np.square(abs(bn(x,y,n)))-(bn(x,y,n)+np.conj(bn(x,y,n)))
                        )
                    )
    return ((-2*np.sqrt(y/x))/c)*(pi/(konda**3))*soma

def TyE(x,y,r0,phi0,z0):
    soma = 0
    for p in range(1,MMAX):
        for n in range(p,NMAX):
            soma += ((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*mpmath.im(
                        (
                            gnmTME(n,p-1,r0,phi0,z0)*np.conj(gnmTME(n,p,r0,phi0,z0)) - gnmTME(n,-p,r0,phi0,z0)*np.conj(gnmTME(n,-p+1,r0,phi0,z0))
                        )*(
                            2*np.square(abs(an(x,y,n)))-(an(x,y,n)+np.conj(an(x,y,n)))
                        )+(
                            gnmTEE(n,p-1,r0,phi0,z0)*np.conj(gnmTEE(n,p,r0,phi0,z0)) - gnmTEE(n,-p,r0,phi0,z0)*np.conj(gnmTEE(n,-p+1,r0,phi0,z0))                        
                        )*(
                            2*np.square(abs(bn(x,y,n)))-(bn(x,y,n)+np.conj(bn(x,y,n)))
                        )
                    )
    return ((-2*np.sqrt(y/x))/c)*(pi/(konda**3))*soma
    
def TzLA(x,y,r0,phi0,z0):
    soma = 0
    for p in range(-MMAX,MMAX):
        if p==0:
            soma = 0
        else:
            for n in range(abs(p),NMAX):
                soma += p*((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*(
                    (abs(gnmTMLA(n, p, r0, phi0, z0))**2)*(mpmath.re(an(x,y,n))-abs(an(x,y,n))**2) + (abs(gnmTELA(n, p, r0, phi0, z0))**2)*(mpmath.re(bn(x,y,n))-abs(bn(x,y,n))**2)                
                    )
    return ((-4*np.sqrt(y/x))/c)*(pi/(konda**3))*soma

def TzLAV(x,y,r0,phi0,z0):
    soma = 0
    for p in range(-MMAX,MMAX):
        if p==0:
            soma = 0
        else:
            for n in range(abs(p),NMAX):
                soma += p*((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*(
                    (abs(gnmTMLAV(n, p, r0, phi0, z0))**2)*(mpmath.re(an(x,y,n))-abs(an(x,y,n))**2) + (abs(gnmTELAV(n, p, r0, phi0, z0))**2)*(mpmath.re(bn(x,y,n))-abs(bn(x,y,n))**2)                
                    )
    return ((-4*np.sqrt(y/x))/c)*(pi/(konda**3))*soma

def TzE(x,y,r0,phi0,z0):
    soma = 0
    for p in range(-MMAX,MMAX):
        if p==0:
            soma = 0
        else:
            for n in range(abs(p),NMAX):
                soma += p*((2*n+1)/(n*(n+1)))*(mpmath.fac(n+p)/mpmath.fac(n-p))*(
                    (abs(gnmTME(n, p, r0, phi0, z0))**2)*(mpmath.re(an(x,y,n))-abs(an(x,y,n))**2) + (abs(gnmTEE(n, p, r0, phi0, z0))**2)*(mpmath.re(bn(x,y,n))-abs(bn(x,y,n))**2)                
                    )
    return ((-4*np.sqrt(y/x))/c)*(pi/(konda**3))*soma

'''|====|====|====|====|====|====|====|'''

'''|====|====|====|Classes|====|====|====|'''
'''class TMLAThread (threading.Thread):
    def __init__(self, name, counter, n, m, r0, phi0, z0):
        threading.Thread.__init__(self)
        self.threadID = counter
        self.name = name
        self.counter = counter
    def run(self):
        # print("Starting " + self.name + self.threadID)
        threadLock.acquire()
        gnmTMLA(n, m, r0, phi0, z0)
        # print("Exiting " + self.name)
        threadLock.release()'''
        
'''threadLock = threading.Lock()'''
'''|====|====|====|====|====|====|====|'''

'''|====|====|====|Verificação das variaveis|====|====|====|'''

'''|====|====|====|====|====|====|====|'''

'''|====|====|====|Codigo principal|====|====|====|'''

def tratar_entrada(entradas, atributo):
    n = len(entradas)
    for i in range (1, n):
        try:
            dado = entradas[i].split("=")
            if dado[0]==atributo:
                resultado=dado[1]            
        except IndexError:
            dado = "Erro"
    return resultado

if __name__=="__main__":
    while True:
        try:
            #tipo = int(input("Type of Frozen Wave: "))
            global usuario_sistema
            usuario_sistema = tratar_entrada(sys.argv, "usuario")
        except ValueError:
            nope()
            continue
        else:
            # print("Tipo=", tipo)
            break
    while True:
        try:
            #tipo = int(input("Type of Frozen Wave: "))
            tipo = int(tratar_entrada(sys.argv, "tipo"))
        except ValueError:
            nope()
            continue
        else:
            # print("Tipo=", tipo)
            break
    checktipo()
    while True:
        try:
            #ordem = int(input("Order of Frozen Wave: "))
            ordem = int(tratar_entrada(sys.argv,"ordem"))
        except ValueError:
            nope()
            continue
        else:
            # print("Ordem=", ordem)
            break
    checkordem()
    while True:
        try:
            #choice = int(input("Option: "))
            choice = int(tratar_entrada(sys.argv,"option"))
        except ValueError:
            nope()
            continue
        else:
            # print("You selected option:", choice)
            break
        
    if choice == 1:
        print("1")
        #step 200
        startall = timer()
        vBJx = []
        vX = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(-10*(spot0*10**6),10*(spot0*10**6),step))
        # print("Vector x in µm:", fstep, "Size:", len(fstep))
        startfor = timer()
        for dx in fstep:
                BJx = mpmath.re(mpmath.besselj(ordem, dx*(10**(-6))*mpmath.re(kphop(0))))
                # print("Jn[",dx,"] = ", BJx)
                endfor = timer()
                # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
                vBJx.append(float(BJx))
                vX.append(float(dx))
        ##axes = plt.gca()
        #axes.set_xlim([-10*(float(spot0)*10**6),10*(float(spot0)*10**6)])
        #ylim = [float(min(vBJx)),float(max(vBJx))]
        ##axes.set_ylim(ylim)
        #plt.plot(vX,vBJx)
        #plt.grid(True)
        #plt.xlabel(''x(µm)')
        #plt.ylabel('Jn(x)')
        #plt.title('Graph x by Jn(x)')
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        #np.savetxt('BesselJ'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'_'+'.json'+'IMOC', np.transpose([vBJx,vX]))            
        titulos = {"x(µm)":"Jn(x)"}
        vetores = dict(zip(vX,vBJx))
        print("prexec json")
        gerar_arquivo_json('BesselJ'+'_'+'r'+str(step)+'.json', vetores, titulos)

    elif choice == 2:
        #step = 0.0025
        startall = timer()
        #plt.plot(range(-NN,NN+1),np.real(testeAp()))
        #plt.grid(True)
        #plt.xlabel('q')
        #plt.ylabel('Re(Aq)')
        #plt.title('Real Values of A by its subindex')
        ## plt.show()
        #plt.plot(range(-NN,NN+1),np.imag(testeAp()))
        #plt.grid(True)
        #plt.xlabel('q')
        #plt.ylabel('Im(Aq)')
        #plt.title('Imaginary Values of A by its subindex')
        ## plt.show()
        #plt.plot(range(-NN,NN+1),np.abs(testeAp()))
        #plt.grid(True)
        #plt.xlabel('q')
        #plt.ylabel('Abs(Aq)')
        #plt.title('Absolute Values of A by its subindex')
        ## plt.show()
        vPsi = []
        vZ = []
        step = float(tratar_entrada(sys.argv,"step"))        
        fstep = list(np.linspace(-2.5*(10**3)*Zmax*L,2.5*(10**3)*Zmax*L,step))
        # print("Vector Z in µm:", fstep, "Size:", len(fstep))
        startfor = timer()
        for dz in fstep:
            Psiz = (np.abs(Psi(localspot,dz*10**(-3),0,0*10**(-4))))**2
            vPsi.append(float(Psiz))
            vZ.append(dz)
        ##axes = plt.gca()
        #axes.set_xlim([-2.5*(10**3)*Zmax*L,2.5*(10**3)*Zmax*L])
        #plt.plot(vZ,vPsi)
        #plt.grid(True)
        #plt.xlabel(r'$z(mm)$', fontsize = 14)
        #plt.ylabel(r'$|\Psi(localspot,z10^{-3},0,0)|^{2}$', fontsize = 14)
        #plt.title(r'$z$ by $|\Psi(localspot,z10^{-3},0,0)|^{2}$',fontsize=16)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"z(mm)":"|Psi(localspot,z*10**(-3),0,0)|²"}
        vetores = dict(zip(vZ,vPsi))
        gerar_arquivo_json('Psi'+'_'+'r'+str(step)+'.json', vetores, titulos)
    elif choice == 3:
        #tipo 8
        #ordem 0
        #step 110
        startall = timer()
        yy = 1.05**2
        vz=[]
        vCprzLA = []
        vCprzLAV = []
        vCprzE = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(L*-0.5*10**3,L*0.5*10**3,step))
        # print("Z's range:", step)
        # print("Vector Z in mm", fstep, "Size:", len(fstep))
        frar = list(np.linspace(-10*spot*10**-6,10*spot*10**-6,200))
        # print("Vector r0 in µm", frar, "Size:", len(frar))
        startfor = timer()
        for dz in fstep:
            aCprzE = CprzE(raiop,yy*raiop,0,0,dz*10**-3)
            # #print("Valor de CprzE para z = ", dz, "     CprzE[",raiop,",", yy*raiop ,", 0, 0,",dz,"*10**-3] = ", aCprzE)
            vCprzE.append(aCprzE)       
            aCprzLA = CprzLA(raiop,yy*raiop,0,0,dz*10**-3)
            # #print("Valor de CprzLA para z = ", dz, "     CprzLA[",raiop,",", yy*raiop ,",0,0,",dz,"*10**-3] = ", aCprzLA)
            vCprzLA.append(aCprzLA)   
            aCprzLAV = CprzLAV(raiop,yy*raiop,0,0,dz*10**-3)
            # #print("Valor de CprzLAV para z = ", dz, "     CprzLAV[",raiop,",", yy*raiop ,",0,0,",dz,"*10**-3] = ", aCprzLAV)
            vCprzLAV.append(aCprzLAV)             
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vz.append(dz)
        ##axes = plt.gca()
        #axes.set_xlim([-10*spot*10**-6,(10*spot*10**-6)])
        #ylim = [float(min(vCprzE)),float(max(vCprzE))]
        ##axes.set_ylim(ylim)
        #plt.plot(vz,vCprzE, linestyle='-', color='black')
        #plt.plot(vz,vCprzLA, linestyle='-', color='blue')
        #plt.plot(vz,vCprzLAV, linestyle='--', color='red')
        #plt.legend(['$C^{E}_{pr,z}$', '$C^{loc}_{pr,z}$', '$C^{locCC}_{pr,z}$'], loc='upper right',fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$z (mm)$',fontsize = 16)
        #plt.ylabel(r'$C^{E}_{pr,z} (m^2)$',fontsize = 16)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 4:
        #tipo 8
        #ordem 0
        #step 110
        startall = timer()
        yy = 1.05**2
        rar = float(tratar_entrada(sys.argv,"step"))
        frar = list(np.linspace(-10*spot*10**-6,10*spot*10**-6,rar))
        # print("r0's range:", step) #range de teste: 0.01
        # print("Vector r0 in µm", frar, "Size:", len(frar))
        vr0=[]
        vCprxLA = []
        vCprxLAV = []
        vCprxE = []
        startfor = timer()
        for dr in frar:
            aCprxE = CprxE(raiop,yy*raiop,dr*10**-6,0,0)
            # #print("Valor de CprxE para r0 = ", dr, "     CprxE[",raiop,",", yy*raiop ,",",dr,"*10**-6, 0, 0] = ", aCprxE)
            vCprxE.append(aCprxE)       
            aCprxLA = CprxLA(raiop,yy*raiop,dr*10**-6,0,0)
            # #print("Valor de CprxLA para r0 = ", dr, "     CprxLA[",raiop,",", yy*raiop ,",",dr,"*10**-6, 0, 0] = ", aCprxLA)
            vCprxLA.append(aCprxLA)  
            aCprxLAV = CprxLAV(raiop,yy*raiop,dr*10**-6,0,0)
            # #print("Valor de CprxLAV para r0 = ", dr, "     CprxLAV[",raiop,",", yy*raiop ,",",dr,"*10**-6, 0, 0] = ", aCprxLAV)
            vCprxLAV.append(aCprxLAV)             
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vr0.append(dr)
        ##axes = plt.gca()
        #axes.set_xlim([L*-0.5*10**3,(L*0.5*10**3)+step])
        ##ylim = [float(min(vBJx)),float(max(vBJx))]
        ###axes.set_ylim(ylim)
        #plt.plot(vr0,vCprxE, linestyle='-', color='black')
        #plt.plot(vr0,vCprxLA, linestyle='-', color='blue')
        #plt.plot(vr0,vCprxLAV, linestyle='--', color='red')
        #plt.legend(['$C^{E}_{pr,z}$', '$C^{loc}_{pr,z}$', '$C^{locCC}_{pr,z}$'], loc='upper right',fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$z (mm)$',fontsize = 16)
        #plt.ylabel(r'$C^{E}_{pr,z} (m^2)$',fontsize = 16)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 5:
        r0 = 0
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]
        # print("===============================================")
        # print("order = 0 e r0 = 0")
        for na in naLista:
            rint(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na

        ordem = 3
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        # print("===============================================")
        # print("ordem = 3 e r0 = 0")
        for na in naLista:
             print(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na
       
        ordem = 0
        r0 = 10*10**(-6)
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        # print("===============================================")
        # print("ordem = 0 e r0 = 10*10^(-6)")
        for na in naLista:
            print(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na, r0
        
        ordem = 3
        r0 = 10*10**(-6)
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        # print("===============================================")
        # print("ordem = 3 e r0 = 10*10^(-6)")
        for na in naLista:
             print(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na, r0
        
        ordem = 2
        r0 = 10*10**(-6)
        z0 = 10**(-3)
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        # print("===============================================")
        # print("ordem = 2, r0 = 10*10^(-6) e z = 10^(-3)")
        for na in naLista:
             print(na, gnmTME(na,ordem + 1,r0,0,z0), gnmTMLA(na,ordem + 1,r0,0,z0), gnmTMLAV(na,ordem + 1,r0,0,z0))
 
        del na,r0,z0,ordem
    elif choice == 6:
        ordem = 0
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]
        # print("===============================================")
        # print("ordem = 0")
        for na in naLista:
             print(na, tau(na, ordem + 1, 0.2), fpi(na, ordem + 1, 0.2))
        
        ordem = 3
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        # print("===============================================")
        # print("ordem = 3")
        for na in naLista:
             print(na, tau(na, ordem + 1, 0.2), fpi(na, ordem + 1, 0.2))
    elif choice == 7:
        yy = 1.01**2
        for n in range(1,11+2):
             print(n, an(lambd, yy*lambd, n), bn(lambd, yy*lambd, n))
        del yy
    elif choice == 8:
        #tipo 1
        #ordem = 0
        #step = 150
        startall = timer()
        ma = 1
        r0 = 0
        phi0 = 0
        z0 = 0
        vn = []
        vgnmTMLA = []
        vgnmTME = []
#        vgnmTMLAV = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(1.001,400,step))
        # print("n range:", step)
        # print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTME = re(gnmTME(dna, ma, r0, phi0, z0))
            vgnmTME.append(float(abgnmTME))     
            abgnmTMLA = re(gnmTMLA(dna, ma, r0, phi0, z0))
            vgnmTMLA.append(float(abgnmTMLA))                  
#            abgnmTMLAV = gnmTMLAV(dna, ma, r0, phi0, z0)
#            vgnmTMLAV.append(float(abgnmTMLAV))            
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        ##axes = plt.gca()
        #axes.set_xlim([0,400+step])
        #ylim = [float(min(vgnmTMLA)*1.2),float(max(vgnmTMLA)*1.2)]
        ##axes.set_ylim(ylim)
        ##plt.ylim(-0.2, 0.6)
        #plt.yticks(np.arange(-0.2,0.55,0.2))
        #plt.xlim(0, 400)
        #plt.xticks(np.arange(0,401,100))
        #plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
##        plt.plot(vn,vgnmTMLAV, linestyle='--', color='blue')
##        plt.legend(['$g^{\pm1,exa}_{n,TM}$', '$g^{\pm1,loc}_{n,TM}$', '$g^{\pm1,locCC}_{n,TM}$'], loc='upper right',fontsize = 16)
        #plt.legend(['$g^{\pm1,exa}_{n,TM}$', '$g^{\pm1,loc}_{n,TM}$'], loc='upper right',fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$n$',fontsize = 16)
        #plt.ylabel(r'$|g^{\pm1}_{n,TM}|$',fontsize = 16)
###        a = plt.axes([.25, 0.65, .25, .25])
#        testvgnmTME = [x*constants.kilo for x in vgnmTME]
#        testvgnmTMLA = [x*constants.kilo for x in vgnmTMLA]
#        testvgnmTMLAV = [x*constants.kilo for x in vgnmTMLAV]
##        plt.plot(vn,testvgnmTME, linestyle='-', color='black')
##        plt.plot(vn,testvgnmTMLA, linestyle='--', color='red')
##        plt.plot(vn,testvgnmTMLAV, linestyle='--', color='blue')
##        plt.grid(True)
##        plt.ylabel(r'$(\times10^-3)$',fontsize = 12)
###        plt.ylim(-4, 4)
##        plt.yticks(np.arange(-4,5,2))
##        plt.xlim(300, 400)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"n":"[gnmTME(n,±1,0,0,0),gnmTMLA(n,±1,0,0,0)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig1'+'_'+'r'+str(step)+'.json', vetores, titulos)
    elif choice == 9:
        #tipo 1
        #ordem 0
        #step 150
        startall = timer()
        na = 1
        ma = -1
        phi0 = 0
        r0 = 0
        vgnmTMLA = []
        vgnmTME = []
        vgnmTMLAV = []
        #step = float(input("r0 range: "))
        step = float(tratar_entrada(sys.argv, "step"))
        fstep = list(np.linspace(0,3*constants.milli,step))
        # print("n range:", step)
        # print("Vector n", fstep, "Size:", len(fstep))
        flin = np.linspace(fstep[0],20*constants.micro,300)
        startfor = timer()
        for dz in flin:
            abgnmTMLA = mpmath.re(gnmTMLA(na, ma, r0, phi0, dz))
            # #print("Valor de gnmTMLA para n = ", dz, " , ", abgnmTMLA)
            vgnmTMLA.append(float(abgnmTMLA))
            #com_vgnmTMLA.append(float(bbgnmTMLA.real))       
            abgnmTME = mpmath.re(gnmTME(na, ma, r0, phi0, dz))
            # #print("Valor de gnmTME para n = ", dz, " , ", abgnmTME)
            vgnmTME.append(float(abgnmTME))
            #com_vgnmTME.append(float(bbgnmTME.real))                 
#            abgnmTMLAV = gnmTMLAV(na, ma, r0, phi0, dz)
#            vgnmTMLAV.append(float(abgnmTMLAV.real))
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
        vgnmTMLA2 = []
        vgnmTME2 = []
        vgnmTMLAV2 = []
        for dz in fstep:
            abgnmTMLA = gnmTMLA(na, ma, r0, phi0, dz)
            # #print("Valor de gnmTMLA para n = ", dz, " , ", abgnmTMLA)
            vgnmTMLA2.append(float(abgnmTMLA.real))
            #com_vgnmTMLA.append(float(bbgnmTMLA.real))       
            abgnmTME = gnmTME(na, ma, r0, phi0, dz)
            # #print("Valor de gnmTME para n = ", dz, " , ", abgnmTME)
            vgnmTME2.append(float(abgnmTME.real))
            #com_vgnmTME.append(float(bbgnmTME.real))                 
#            abgnmTMLAV = gnmTMLAV(na, ma, r0, phi0, dz)
#            vgnmTMLAV2.append(float(abgnmTMLAV.real))
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
        #fig = plt.figure(figsize=(7,5.25))
        ##axes = plt.gca()
        #axes.set_xlim([0,20*constants.micro+0.5*constants.micro])
        #ticks = axes.get_xticks()*10**6
        #axes.set_xticklabels(ticks)
        #ylim = [float(min(vgnmTMLA)*1.05),float(max(vgnmTMLA)*1.05)]
        ##axes.set_ylim(ylim)
        #plt.plot(flin,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(flin,vgnmTMLA, linestyle='--', color='red')
##        plt.plot(flin,vgnmTMLAV, linestyle='--', color='blue')
##        plt.legend(['$g^{\pm1,exa}_{n,TM}$', '$g^{\pm1,loc}_{n,TM}$', '$g^{\pm1,locCC}_{n,TM}$'], loc='upper right', fontsize = 16)
        #plt.legend(['$g^{\pm1,exa}_{n,TM}$', '$g^{\pm1,loc}_{n,TM}$'], loc='upper right', fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$z_0 (\mu m)$', fontsize = 16)
        #plt.ylabel(r'$|g^{\pm1}_{n,TM}|$', fontsize = 16)
        ## plt.show()
        #fig = plt.figure(figsize=(7,5.25))
        ##axes2 = plt.gca()
        #plt.plot(fstep,vgnmTME2, linestyle='-', color='black', linewidth=1.5, markerfacecolor='w')
        #plt.plot(fstep,vgnmTMLA2, linestyle='--', color='red', markerfacecolor='w')
##        plt.plot(fstep,vgnmTMLAV2, linestyle='--', color='blue', markerfacecolor='w')
        #plt.grid(True)
        #plt.xlabel(r'$z_0 (mm)$', fontsize = 16)
        ##plt.ylim(-0.6,0.6)
        #plt.yticks(np.arange(-0.6,0.59,0.2))
        #plt.xlim(0.0,0.0025)
        #ticks = axes.get_xticks()*10**5
        #axes2.set_xticklabels(ticks)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"z":"[gnmTM(1,±1,0,0,z),gnmTM(1,±1,0,0,z)]"}
        vetores = dict(zip(flin,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig2'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.json', vetores, titulos)
        titulos2 = {"z":"[gnmTM(1,±1,0,0,z),gnmTM(1,±1,0,0,z)]"}
        vetores2 = dict(zip(flin,zip(vgnmTME2,vgnmTMLA2)))
        gerar_arquivo_json('subplot'+'_'+'gnm'+'_'+'fig2'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.json', vetores2, titulos2)
    elif choice == 10:
        #tipo = 4
        #ordem = 4
        #step = 150
        startall = timer()
        ma = 3
        r0 = 0
        phi0 = 0
        z0 = 0
        vn = []
        vgnmTMLA = []
        vgnmTME = []
        vgnmTMLAV = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(1.001,200,step))
        # print("n range:", step)
        # print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTMLA = abs(gnmTMLA(dna, ma, r0, phi0, z0))
            vgnmTMLA.append(float(abgnmTMLA))            
            abgnmTME = abs(gnmTME(dna, ma, r0, phi0, z0))
            vgnmTME.append(float(abgnmTME))
#            abgnmTMLAV = abs(gnmTMLAV(dna, ma, r0, phi0, z0))
#            vgnmTMLAV.append(float(abgnmTMLA))
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        ##axes = plt.gca()
        #axes.set_xlim([-8,200])
        testvgnmTME = [x*10**4 for x in vgnmTME]
        testvgnmTMLA = [x*10**4 for x in vgnmTMLA]
#        testvgnmTMLAV = [x*10**4 for x in vgnmTMLAV]
        newvgnmTME = [x*10**5 for x in vgnmTME]
        newvgnmTMLA = [x*10**5 for x in vgnmTMLA]
#        newvgnmTMLAV = [x*10**5 for x in vgnmTMLAV]        
        #ylim = [-0.1,float(max(testvgnmTMLA)*1.05)]
        ##axes.set_ylim(ylim)
        #plt.plot(vn,testvgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(vn,testvgnmTMLA, linestyle='--', color='red')
##        plt.plot(vn,testvgnmTMLAV, linestyle='--', color='blue')
##        plt.legend(['$g^{\pm3,exa}_{n,TM}$', '$g^{\pm3,loc}_{n,TM}$', '$g^{\pm3,locCC}_{n,TM}$'], loc='upper right', fontsize = 16)
        #plt.legend(['$g^{\pm3,exa}_{n,TM}$', '$g^{\pm3,loc}_{n,TM}$'], loc='upper right', fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$n$',fontsize = 16)
        #plt.ylabel(r'$|g^{\pm3}_{n,TM}(\times10^-4)|$', fontsize = 16)
        ##a = plt.axes([0.35, 0.62, .25, .25])
        #plt.plot(vn,newvgnmTME, linestyle='-', color='black')
        #plt.plot(vn,newvgnmTMLA, linestyle='--', color='red')
##        plt.plot(vn,newvgnmTMLAV, linestyle='--', color='blue')
        #plt.grid(True)
        #plt.ylabel(r'$(\times10^-5)$',fontsize = 12)
        ##plt.ylim(0,0.8)
        #plt.yticks(np.arange(0,0.71,0.1))
        #plt.xlim(100, 200)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"n":"[gnmTME(n,±1,0,0,0),gnmTMLA(n,±1,0,0,0)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig3'+'_'+'r'+str(step)+'.json', vetores, titulos)
    elif choice == 11:
        #tipo = 5
        #ordem = 3
        #step = 150
        startall = timer()
        ma = 2
        r0 = 0
        phi0 = 0
        z0 = 0
        vn = []
        vgnmTMLA = []
        vgnmTME = []
        vgnmTMLAV = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(1.001,200,step))
        # print("n range:", step)
        # print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTMLA = abs(gnmTMLA(dna, ma, r0, phi0, z0))
            vgnmTMLA.append(float(abgnmTMLA)*10**2)            
            abgnmTME = abs(gnmTME(dna, ma, r0, phi0, z0))
            vgnmTME.append(float(abgnmTME)*10**2)
#            abgnmTMLAV = abs(gnmTMLAV(dna, ma, r0, phi0, z0))
#            vgnmTMLAV.append(float(abgnmTMLAV)*10**2)
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        ##axes = plt.gca()
        #axes.set_xlim([-8,200])
        #ylim = [-0.025,float(max(vgnmTME)*1.05)]
        ##axes.set_ylim(ylim)
        #plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
##        plt.plot(vn,vgnmTMLAV, linestyle='--', color='blue')
##        plt.legend(['$\mid g^{\pm2,exa}_{n,TM} \mid$', '$\mid g^{\pm2,loc}_{n,TM} \mid$', '$\mid g^{\pm2,locCC}_{n,TM} \mid$'], loc='upper right', fontsize = 16)
        #plt.legend(['$\mid g^{\pm2,exa}_{n,TM} \mid$', '$\mid g^{\pm2,loc}_{n,TM} \mid$'], loc='upper right', fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$n$',fontsize = 16)
        #plt.ylabel(r'$\mid g^{\pm2}_{n,TM} \mid(\times10^-2)$', fontsize = 16)
        ##a = plt.axes([0.31, 0.62, .25, .25])
        #plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
##        plt.plot(vn,vgnmTMLAV, linestyle='--', color='blue')
        #plt.grid(True)
        #plt.ylabel(r'$(\times10^-2)$',fontsize = 12)
        ##plt.ylim(0,0.08)
        #plt.yticks(np.arange(0,0.08,0.02))
        #plt.xlim(100, 200)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"n":"[gnmTME(n,±2,0,0,0),gnmTMLA(n,±2,0,0,0)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig4'+'_'+'r'+str(step)+'.json', vetores, titulos)
    elif choice == 12:
        #tipo 1
        #ordem = 0
        #step = 150
        startall = timer()
        ma = 0
        r0 = 50*constants.micro
        phi0 = 0
        z0 = 0.3*10**-3
        vn = []
        vgnmTME = []
        vgnmTMLA = []
        vgnmTMLAV = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(0,1000,step))
        # print("n range:", step)
        # print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTMLA = abs(gnmTMLA(dna, ma, r0, phi0, z0))
            # #print("Valor de |gnmTMLA| para n = ", dna, "   |gnmTMLA[",dna,",2,0,0,0]| = ", abgnmTMLA)
            vgnmTMLA.append(float(abgnmTMLA))            
            abgnmTME = abs(gnmTME(dna, ma, r0, phi0, z0))
            # #print("Valor de |gnmTME| para n = ", dna, "   |gnmTME[",dna,",2,0,0,0]| = ", abgnmTME)
            vgnmTME.append(float(abgnmTME))
#            abgnmTMLAV = abs(gnmTMLAV(dna, ma, r0, phi0, z0))
#            vgnmTMLAV.append(float(abgnmTMLAV))            
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        ##axes = plt.gca()
        #axes.set_xlim([-1,1001])
        #ylim = [0,float(max(sympify(vgnmTMLA))*1.2)]
        ##axes.set_ylim(ylim)
        xlin = np.linspace(vn[0],vn[-1],300)
        smooth_vgnmTME = spline(vn,vgnmTME,xlin)
        smooth_vgnmTMLA = spline(vn,vgnmTMLA,xlin)
#        smooth_vgnmTMLAV = spline(vn,vgnmTMLAV,xlin)
        #plt.plot(xlin,smooth_vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(xlin,smooth_vgnmTMLA, linestyle='--', color='red')
##        plt.plot(xlin,smooth_vgnmTMLAV, linestyle='--', color='blue')
##        plt.legend(['$\mid g^{0,exa}_{n,TM} \mid$', '$\mid g^{0,loc}_{n,TM} \mid$', '$\mid g^{0,locCC}_{n,TM} \mid$'], loc='upper left', fontsize = 16)
        #plt.legend(['$\mid g^{0,exa}_{n,TM} \mid$', '$\mid g^{0,loc}_{n,TM} \mid$'], loc='upper left', fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$n$', fontsize = 16)
        #plt.ylabel(r'$\mid g^{0}_{n,TM}\mid$', fontsize = 16)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"n":"[gnmTME(n,0,0.00005,0,0.0003),gnmTMLA(n,0,0.00005,0,0.0003)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig5'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.json', vetores, titulos)
    elif choice == 13:
        startall = timer()
        #tipo 1
        #ordem 0
        #step 150
        na = 1
        ma = 0
        phi0 = 0
        z0 = 0.3*10**-3
        vn = []
        vgnmTMLA = []
#        vgnmTMLAV = []
        vgnmTME = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(0,50*constants.micro,step))
        # print("r0 range:", step)
        # print("Vector r0", fstep, "Size:", len(fstep))
        startfor = timer()
        for dr in fstep:
            abgnmTMLA = abs(gnmTMLA(na, ma, dr, phi0, z0))
            # #print("Valor de |gnmTMLA| para n = ", dr, "   |gnmTMLA[200,0,",dr,",0,0]| = ", abgnmTMLA)
            vgnmTMLA.append(float(abgnmTMLA.real))     
            abgnmTME = abs(gnmTME(na, ma, dr, phi0, z0))
            # #print("Valor de |gnmTME| para n = ", dr, "   |gnmTME[200,0,",dr,",0,0]| = ", abgnmTME)
            vgnmTME.append(float(abgnmTME.real))     
#            abgnmTMLAV = abs(gnmTMLAV(na, ma, dr, phi0, z0))
# #            #print("Valor de |gnmTME| para n = ", dr, "   |gnmTME[200,0,",dr,",0,0]| = ", abgnmTME)
#            vgnmTMLAV.append(float(abgnmTMLAV.real))       
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dr)
        ##axes = plt.gca()
        #axes.set_xlim([-1*constants.micro,50*constants.micro])
        #ticks = axes.get_xticks()*10**6
        #axes.set_xticklabels(ticks)
        #ylim = [-0.0001,float(max(sympify(vgnmTMLA))*1.15)]
        ##axes.set_ylim(ylim)
        xlin = np.linspace(vn[0],vn[-1],300)
        smooth_vgnmTME = spline(vn,vgnmTME,xlin)
        smooth_vgnmTMLA = spline(vn,vgnmTMLA,xlin)
#        smooth_vgnmTMLAV = spline(vn,vgnmTMLAV,xlin)
        #plt.plot(xlin,smooth_vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(xlin,smooth_vgnmTMLA, linestyle='--', color='red')
##        plt.plot(xlin,smooth_vgnmTMLAV, linestyle='--', color='blue')
        #plt.legend(['$\mid g^{0,exa}_{1,TM} \mid$', '$\mid g^{0,loc}_{1,TM} \mid$'], loc='upper right', fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$\rho_0 (\mu m)$', fontsize = 16)
        #plt.ylabel(r'$\mid g^{0}_{1,TM}\mid$', fontsize = 16)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"r":"[gnmTME(1,0,r,0,0.0003),gnmTMLA(1,0,r,0,0.0003)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig6'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.json', vetores, titulos)
    elif choice == 14:
        startall = timer()
        #tipo 1
        #ordem 0
        #step 150
        na = 200
        ma = 0
        phi0 = 0
        z0 = 0.3*10**-3
        vn = []
        vgnmTMLA = []
        vgnmTMLAV = []
        vgnmTME = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(0,50*constants.micro,step))
        # print("r0 range:", step)
        # print("Vector r0", fstep, "Size:", len(fstep))
        startfor = timer()
        for dr in fstep:
            abgnmTMLA = abs(gnmTMLA(na, ma, dr, phi0, z0))
            # #print("Valor de |gnmTMLA| para n = ", dr, "   |gnmTMLA[200,0,",dr,",0,0]| = ", abgnmTMLA)
            vgnmTMLA.append(float(abgnmTMLA.real))     
            abgnmTME = abs(gnmTME(na, ma, dr, phi0, z0))
            # #print("Valor de |gnmTME| para n = ", dr, "   |gnmTME[200,0,",dr,",0,0]| = ", abgnmTME)
            vgnmTME.append(float(abgnmTME.real))     
#            abgnmTMLAV = abs(gnmTMLAV(na, ma, dr, phi0, z0))
# #            #print("Valor de |gnmTME| para n = ", dr, "   |gnmTME[200,0,",dr,",0,0]| = ", abgnmTME)
#            vgnmTMLAV.append(float(abgnmTMLAV.real))       
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dr)
        ##axes = plt.gca()
        #axes.set_xlim([-1*constants.micro,50*constants.micro*1.02])
        #ticks = axes.get_xticks()*10**6
        #axes.set_xticklabels(ticks)
        #ylim = [-0.01,float(max(sympify(vgnmTMLA))*1.1)]
        ##axes.set_ylim(ylim)
        xlin = np.linspace(vn[0],vn[-1],300)
        smooth_vgnmTME = spline(vn,vgnmTME,xlin)
        smooth_vgnmTMLA = spline(vn,vgnmTMLA,xlin)
#        smooth_vgnmTMLAV = spline(vn,vgnmTMLAV,xlin)
        #plt.plot(xlin,smooth_vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(xlin,smooth_vgnmTMLA, linestyle='--', color='red')
##        plt.plot(xlin,smooth_vgnmTMLAV, linestyle='--', color='blue')
##        plt.legend(['$\mid g^{0,exa}_{200,TM} \mid$', '$\mid g^{0,loc}_{200,TM} \mid$', '$\mid g^{0,locCC}_{200,TM} \mid$'], loc='upper left', fontsize = 16)
        #plt.legend(['$\mid g^{0,exa}_{200,TM} \mid$', '$\mid g^{0,loc}_{200,TM} \mid$'], loc='upper left', fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$\rho_0 (\mu m)$', fontsize = 16)
        #plt.ylabel(r'$\mid g^{0}_{200,TM}\mid$', fontsize = 16)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"r":"[gnmTME(200,0,r,0,0.0003),gnmTMLA(200,0,r,0,0.0003)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig7'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.json', vetores, titulos)
    elif choice == 15:
        vgnmTME = []
        nlist = [1,2,5,10,25,50,100,200]
        # print("===============================================")
        # print("order = 0 e r0 = 0")
        
        #Sequential Method
        starttask1 = timer()
        for na in nlist:
            gnmTME_val = gnmTME(na,ordem + 1,0,0,0)
            vgnmTME.append(gnmTME_val)
        # print(vgnmTME)
        endtask1 = timer()
        
        #Multiprocessing Method
        starttask2 = timer()   
        p = Pool(processes=4)
        results = [p.apply_async(gnmTME, args=(na,1,0,0,0,)) for na in nlist]
        output = [p.get() for p in results]
        p.join()
        # print(output)
        endtask2 = timer()
        
        mtvgnmTME = []
        #Multithread Method
        starttask3 = timer()
        t = ThreadPool(processes=4)
        mtvgnmTME = t.map(p_gnmTME, nlist)       
        t.close()
        t.join()
        # print(mtvgnmTME)
        endtask3 = timer()
        
        #Analysing Results
        rawtask = (endtask1 - starttask1)/60
        mprocesstask = (endtask2 - starttask2 )/60
        mthreadtask = (endtask3 - starttask3 )/60
          
        # print("Raw Task Performance Time:", rawtask, "minutes")
        # print("MultiProcessing Task Performance Time:", mprocesstask, "minutes")
        # print("MultiThread Task Performance Time:", mthreadtask, "minutes")
        # print("MultiProcessing Task Performance Time Redution:", ((rawtask - mprocesstask) / rawtask )*100, "%")
        # print("MultiThread Task Performance Time Redution:", ((rawtask - mthreadtask) / rawtask )*100, "%")
    elif choice == 16:
        zlist = list(np.linspace(-0.25,0.25,200))
        
        
        #Sequential Method
        starttask1 = timer()
        vPsi = []
        for dz in zlist:
            Psiz = Psi(localspot,dz*10**(-3),0,0)
            vPsi.append(Psiz)
        # print(vPsi)
        endtask1 = timer()
        
        #Multiprocessing Method
        starttask2 = timer()   
        p = Pool(processes=4)
        """resultsPsi = [p.apply_async(Psi, args=(localspot,dz*10**(-3),0,0,)) for dz in zlist]
        mpvPsi = [p.get() for p in resultsPsi]
        p.join()
        # print(mpvPsi)"""
        def nPsi(z,rho,t,z0):
            return Psi(rho,z,t,z0)
        partialPsi = partial(nPsi, rho=localspot, t=0, z0=0)
        mpvPsi = p.map(partialPsi, zlist)
        # print(result_list)
        endtask2 = timer()
        
        mtvPsi = []
        #Multithread Method
        starttask3 = timer()
        t = ThreadPool(processes=4)
        mtvPsi = t.map(partial(Psiz, rho=localspot, t=0, z0=0), zlist)       
        t.close()
        t.join()
        # print(mtvPsi)
        endtask3 = timer()
        
        #Analysing Results
        rawtask = (endtask1 - starttask1)/60
        mprocesstask = (endtask2 - starttask2 )/60
        mthreadtask = (endtask3 - starttask3 )/60
        
        # print("Raw Task Performance Time:", rawtask, "minutes")
        # print("MultiProcessing Task Performance Time:", mprocesstask, "minutes")
        # print("MultiThread Task Performance Time:", mthreadtask, "minutes")
        # print("MultiProcessing Task Performance Time Redution:", ((rawtask - mprocesstask) / rawtask )*100, "%")
        # print("MultiThread Task Performance Time Redution:", ((rawtask - mthreadtask) / rawtask )*100, "%")    
    elif choice == 17:   
        #tipo = 5
        #ordem = 3
        #step = 1.5
        startall = timer()
        ma = 2
        r0 = 0
        phi0 = 0
        z0 = 0
        vn = []
        vgnmTMLA = []
        vgnmTME = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = np.arange(1.001,200+step,step)
        # print("n range:", step)
        # print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTMLA = abs(gnmTMLA(dna, ma, r0, phi0, z0))
            vgnmTMLA.append(float(abgnmTMLA.real)*10**2)            
            abgnmTME = abs(gnmTME(dna, ma, r0, phi0, z0))
            vgnmTME.append(float(abgnmTME.real)*10**2)
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        ##axes = plt.gca()
        #axes.set_xlim([-8,200*1.01+step])
        #ylim = [-0.025,float(max(vgnmTME)*1.05)]
        ##axes.set_ylim(ylim)
        #plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
        #plt.legend(['$\mid g^{\pm2,exa}_{n,TM} \mid$', '$\mid g^{\pm2,loc}_{n,TM} \mid$'], loc='upper right', fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$n$',fontsize = 16)
        #plt.ylabel(r'$\mid g^{\pm2}_{n,TM} \mid(\times10^-2)$', fontsize = 16)
        ##a = plt.axes([0.31, 0.62, .25, .25])
        #plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
        #plt.grid(True)
        #plt.ylabel(r'$(\times10^-2)$',fontsize = 12)
        ##plt.ylim(0,0.08)
        #plt.yticks(np.arange(0,0.08,0.02))
        #plt.xlim(100, 200)
        ## plt.show()
        titulos = {"n":"[gnmTME(n,±1,0,0,0),gnmTMLA(n,±1,0,0,0)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig2'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.json', vetores, titulos)
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")    
    elif choice == 18:
        #tipo 1
        #ordem = 0
        #step = 200
        startall = timer()
        ma = 1
        r0 = 0
        phi0 = 0
        z0 = 0
        vn = []
        vgnmTMLA = []
        vgnmTME = []
        vgnmTMLAV = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = list(np.linspace(1.001,400,step))
        # print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTME = gnmTME(dna, ma, r0, phi0, z0)
            vgnmTME.append(float(abgnmTME.real))     
            abgnmTMLA = gnmTMLA(dna, ma, r0, phi0, z0)
            vgnmTMLA.append(float(abgnmTMLA.real))      
            endfor = timer()
            # print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        ##axes = plt.gca()
        #axes.set_xlim([0,400])
        #ylim = [float(min(vgnmTMLA)*1.2),float(max(vgnmTMLA)*1.2)]
        ##axes.set_ylim(ylim)
        ##plt.ylim(-0.2, 0.6)
        #plt.yticks(np.arange(-0.2,0.55,0.2))
        #plt.xlim(0, 400)
        #plt.xticks(np.arange(0,401,100))
        #plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        #plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
        #plt.legend(['$g^{\pm1,exa}_{n,TM}$', '$g^{\pm1,loc}_{n,TM}$'], loc='upper right',fontsize = 16)
        #plt.grid(True)
        #plt.xlabel(r'$n$',fontsize = 16)
        #plt.ylabel(r'$g^{\pm1}_{n,TM}$',fontsize = 16)
        ## plt.show()
        endall = timer()
        # print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 0:
        # print("Ok, good bye.")
        time.sleep(1)
        raise SystemExit
    else:
        nope()
        raise SystemExit
