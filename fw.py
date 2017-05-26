# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:15:18 2016
@project: Simulações IMOC
@author: Ivan
"""

ver = 1.3

import numpy as np
import pylab
import sys
from scipy import special as scisp
from scipy.integrate import quad
from scipy import constants
import matplotlib.pyplot as plt
from sympy import mpmath as mpmath
from sympy import sympify
from timeit import default_timer as timer
from datetime import date
import time
import json

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
    print('!!!!Error!!!')
    time.sleep(0.5)
    print(np.random.choice(stuff))
    time.sleep(0.7)
    
def krodel(i,j):
    if i == j:
        return 1
    elif i != j:
        return 0
    else:
        return nope()

def menu():
    print("Choose one option:")
    print("Type '1' to test BesseJ for ordem and x*kphop(0)")
    print("Type '2' to test Abs[Phi]² for ")
    print("Type '3' to test gnmTME, gnmTMLA and gnmTMLAV graph")
    print("Type '4' to test Cprz and x graphs in E, LA and LAV approachs")    
    print("Type '5' to test gnmTME, gnmTMLA and gnmTMLAV numerically")
    print("Type '6' to test tau and pi")
    print("Type '7' to test an and bn")
    print("Type '8' to make Fig.1")    
    print("Type '9' to make Fig.2")   
    print("Type '10' to make Fig.3")
    print("Type '11' to make Fig.4")
    print("Type '0' to exit.")
    
'''Função que retorna a resolução de uma integral complexa. É feita a partir da
resolução da integral separadamente pela sua parte real e complexa sem o i(ou 1j
na linguagem do Python e depois estas são somadas e é acrescentado o 1j para 
tornar a parte complexa de fato complexa. As integrais são feitas a partir do
input de valores na função quad do scipy.integrate que as resolve por metodo de
quadratura. Foi escrita por "dr jimbob" em: 
http://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers'''
def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]

def gerar_arquivo_json(nome_arquivo, matriz, titulos):
    with open(nome_arquivo, "w", encoding='utf-8') as outfile: 
        json.dump([[titulos, matriz]], outfile, ensure_ascii=False, sort_keys=True, indent=1, separators=(',', ':'))

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

#comrpimento de onda no vácuo
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
    elif tipo == 2: #FW2
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
    elif tipo == 4:
        L = 1*10**(-3)
        NN = 15
        A = 0.9879999999999
        f = lambda _: 1
        Zmax = 0.1
    elif tipo == 5:
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
    spot0 = 2.405/mpmath.re(kphop(0))

    #Raio mínimo da abertura física para geração eficiente da Frozen Wave
    R = 2*L*(mpmath.re(kphop(-NN))/mpmath.re(betap(-NN)))
'''|====|====|====|====|====|====|====|'''



'''|====|====|====|Linha AP|====|====|====|'''

'''cx = "co"
while cx != "q":
   #print("Choose one type of coeficient:")
   #print("Type 'co' to calculate using a constant coeficient")
   #print("Type 'z' to calculate using a 0")
   #print("Type 'q' to quit")
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
    return [Ap(p) for p in range(-NN,NN+1)]
            

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
    
if tipo==0:
    def gnmTMLAI(n,m,r0,phi0,z0):
        soma = 0
        for p in range(-NN,NN+1):
            soma += (1/2)*Znm(n,m)*((-1j)**ordem)*testeAp()[p+NN]*(
                CoefA1*mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
                +CoefA2*mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(- 1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
                )*mpmath.exp(1j*betap(p)*z0)
        return soma        
    def gnmTELAI(n,m,r0,phi0,z0):
        soma = 0
        for p in range(-NN,NN+1):
            soma += (1/2)*Znm(n,m)*((-1j)**ordem)*testeAp()[p+NN]*(
                CoefB1*mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
                -CoefB2*mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(-1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
                )*mpmath.exp(1j*betap(p)*z0)
        return soma
else:
    def gnmTMLAI(n,m,r0,phi0,z0):
        soma = 0
        for p in range(-NN,NN+1):
            soma += (1/2)*Znm(n,m)*((-1j)**ordem)*testeAp()[p+NN]*(
                CoefA1*mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
                +CoefA2*mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(- 1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
                )*mpmath.exp(1j*betap(p)*z0)
        return soma
    def gnmTELAI(n,m,r0,phi0,z0):
        soma = 0
        for p in range(-NN,NN+1):
            soma += (1/2)*Znm(n,m)*((-1j)**ordem)*testeAp()[p+NN]*(
                CoefB1*mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
                -CoefB2*mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(- 1 + m - ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
                )*mpmath.exp(1j*betap(p)*z0)
        return soma

'''|====|====|====|====|====|====|====|'''



'''|====|====|====|BSCs USING THE ILA, BUT FOR CIRCULARLY SYMMETRIC FWs. 
THIS IS BETTER FOR COMPARISON WITH THE EXACT ONE, BUT ARE NOT THE ONE USED IN PREVIOUS WORKS.
INDEED, PREVIOUS WORKS RELIED ON THE BSCs SHOWN ABOVE|====|====|====|'''

if tipo==0:
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
            -mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
            )+(1-mpmath.cos(axicon(p)))*(
            -mpmath.besselj(1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(-1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(1+m)*phi0)
            +mpmath.besselj(-1+m,Rloc(n)*mpmath.sin(axicon(p)))*mpmath.besselj(1+m-ordem,r0*konda*mpmath.sin(axicon(p)))*np.exp(-1j*(-1+m)*phi0)
            ))*mpmath.exp(1j*betap(p)*z0)
        return soma
else:
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

if tipo==0:
    def gnmTMEI(n,m,r0,phi0,z0):
        soma = 0
        for p in range(-NN,NN+1):
            soma += testeAp()[p+NN]*g(p)*(
            (1j**(ordem-m+1))*np.exp(1j*(ordem - m + 1)*phi0)*mpmath.besselj(ordem-m+1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))+m*fpi(n,m,axicon(p)))
            +(1j**(ordem-m-1))*np.exp(1j*(ordem - m - 1)*phi0)*mpmath.besselj(ordem-m-1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))-m*fpi(n,m,axicon(p)))
            )*mpmath.exp(1j*betap(p)*z0)
        return (-(-1)**((m-abs(m))/2))*(mpmath.fac(n-m)/mpmath.fac(n+abs(m)))*soma        
    def gnmTEEI(n,m,r0,phi0,z0):
        soma = 0
        for p in range(-NN,NN+1):
            soma += testeAp()[p+NN]*g(p)*(
            (1j**(ordem-m+1))*np.exp(1j*(ordem - m + 1)*phi0)*mpmath.besselj(ordem-m+1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))+m*fpi(n,m,axicon(p)))
            -(1j**(ordem-m-1))*np.exp(1j*(ordem - m - 1)*phi0)*mpmath.besselj(ordem-m-1,r0*konda*mpmath.sin(axicon(p)))*(tau(n,m,axicon(p))-m*fpi(n,m,axicon(p)))
            )*mpmath.exp(1j*betap(p)*z0)
        return (1j*(-1)**((m-abs(m))/2))*(mpmath.fac(n-m)/mpmath.fac(n+abs(m)))*soma 
else:
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


'''|====|====|====|Verificação das variaveis|====|====|====|'''
def checkvar():
    print("No de BBs para Frozen Wave = ", (2*NN)+1)
    print("Número da onda em relação a frequência escolhida (m^-1) = ", konda)
    print("Coeficiente A (Parâmetro Q do MICHEL)", A)
    print("Spot do feixe de Bessel central (µm) = ", spot0*10**6)
    print("Ponto radial do centro do spot principal = ", localspot)
    print("Raio mínimo de abertura (mm) = ", R*10**3)
    print("Raio da partícula", raiop)
    print("....Testando betam.....")
    for p in range(1,2*NN+2):
        print("Numeros de onda longitudinal para p = ", p, "   beta[",p,"](m^-1) = ", betap(p-NN-1))
    print("....Testando kphom.....")
    for p in range(1,2*NN+2):
        print("Numeros de onda radiais para p = ", p, "   kphop[",p,"](m^-1) = ", kphop(p-NN-1))
    print("....Testando Axicon.....")
    for p in range(1,2*NN+2):
        print("Angulo do axicon para p = ", p, "   axicon[",p,"](graus) = ", axicon(p-NN-1)*(180/pi))
    print("....Testando Am.....")
    print("\n".join(map(str,testeAp())))
    print("....Checando valores dos gnms....")
    print("[na = 1,ma = 1,r0 = 0,phi0 = 0,z0 = 0*10^(-5)]")
    print("gnmTMLA[na,ma,r0,phi0,z0] = ",gnmTMLA(1,1,0,0,0*10**(-5)))
    print("gnmTELA[na,ma,r0,phi0,z0] = ",gnmTELA(1,1,0,0,0*10**(-5)))
    print("gnmTMLAV[na,ma,r0,phi0,z0] = ",gnmTMLAV(1,1,0,0,0*10**(-5)))
    print("gnmTELAV[na,ma,r0,phi0,z0] = ",gnmTELAV(1,1,0,0,0*10**(-5)))
    print("gnmTME[na,ma,r0,phi0,z0] = ",gnmTME(1,1,0,0,0*10**(-5)))
    print("gnmTEE[na,ma,r0,phi0,z0] = ",gnmTEE(1,1,0,0,0*10**(-5)))
    #print("....Exemplos de erros percentuais de gnmTM e gnmTE")
    #if abs(gnmTME(1,1,0,0,0*10**(-5)))==0:
    #    print("Whoops, division by 0")
    #else:
    #    print(100*abs((abs(gnmTME(1,1,0,0,0*10**(-5)))) - abs(gnmTMLA(1,1,0,0,0*10**(-5))))/abs(gnmTME(1,1,0,0,0*10**(-5))))
    #if abs(gnmTEE(1,1,0,0,0*10**(-5)))==0:
    #    print("Whoops, division by 0")
    #else:
    #    print(100*abs((abs(gnmTEE(1,1,0,0,0*10**(-5)))) - abs(gnmTELA(1,1,0,0,0*10**(-5))))/abs(gnmTEE(1,1,0,0,0*10**(-5))))
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
            tipo = int(tratar_entrada(sys.argv, "tipo"))
        except ValueError:
            nope()
            continue
        else:
            print("You selected option:", tipo)
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
            print("You selected option:", ordem)
            break
    checkordem()
    checkvar()
    while True:
        try:
            menu()
            #choice = int(input("Option: "))
            choice = int(tratar_entrada(sys.argv,"option"))
        except ValueError:
            nope()
            continue
        else:
            print("You selected option:", choice)
            break
        
    if choice == 1:
        startall = timer()
        vBJx = []
        vX = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = np.arange(-10*(spot0*10**6),10*(spot0*10**6)+step,step)
        print("x range:", step) #range de teste: 0.005
        print("Vector x in µm:", fstep, "Size:", len(fstep))
        startfor = timer()
        for dx in fstep:
                BJx = mpmath.re(mpmath.besselj(ordem, dx*(10**(-6))*mpmath.re(kphop(0))))
                print("Jn[",dx,"] = ", BJx)
                endfor = timer()
                print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
                vBJx.append(float(BJx))
                vX.append(float(dx))
        axes = plt.gca()
        axes.set_xlim([-10*(float(spot0)*10**6),10*(float(spot0)*10**6)+step])
        ylim = [float(min(vBJx)),float(max(vBJx))]
        axes.set_ylim(ylim)
        plt.plot(vX,vBJx)
        plt.grid(True)
        plt.xlabel('x(µm)')
        plt.ylabel('Jn(x)')
        plt.title('Graph x by Jn(x)')
        plt.show()
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        #np.savetxt('BesselJ'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'_'+'.txt'+'IMOC', np.transpose([vBJx,vX]))            
        titulos = {"x(µm)":"Jn(x)"}
        vetores = dict(zip(vX,vBJx))
        gerar_arquivo_json('BesselJ'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.txt', vetores, titulos)
    elif choice == 2:
        startall = timer()
        plt.plot(range(-NN,NN+1),np.real(testeAp()))
        plt.grid(True)
        plt.xlabel('q')
        plt.ylabel('Re(Aq)')
        plt.title('Real Values of A by its subindex')
        plt.show()
        plt.plot(range(-NN,NN+1),np.imag(testeAp()))
        plt.grid(True)
        plt.xlabel('q')
        plt.ylabel('Im(Aq)')
        plt.title('Imaginary Values of A by its subindex')
        plt.show()
        plt.plot(range(-NN,NN+1),np.abs(testeAp()))
        plt.grid(True)
        plt.xlabel('q')
        plt.ylabel('Abs(Aq)')
        plt.title('Absolute Values of A by its subindex')
        plt.show()
        vPsi = []
        vZ = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = np.arange(-2.5*(10**3)*Zmax*L,2.5*(10**3)*Zmax*L+step,step)
        print("Z range:", step) #range recomendada 0.005
        print("Vector Z in µm:", fstep, "Size:", len(fstep))
        startfor = timer()
        for dz in fstep:
            Psiz = (np.abs(Psi(localspot,dz*10**(-3),0,0*10**(-4))))**2
            print("Value of |Psi|² for z = ", dz, "   |Psi[",localspot,dz*10**(-3),",0,0]|² = ", Psiz)
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vPsi.append(float(Psiz))
            vZ.append(dz)
        axes = plt.gca()
        axes.set_xlim([-2.5*(10**3)*Zmax*L,2.5*(10**3)*Zmax*L+step])
        plt.plot(vZ,vPsi)
        plt.grid(True)
        plt.xlabel('z(mm)')
        plt.ylabel('|Psi(localspot,z*10**(-3),0,0)|²')
        plt.title('Graph z by |Psi(localspot,z*10**(-3),0,0)|²')
        plt.show()
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"z(mm)":"|Psi(localspot,z*10**(-3),0,0)|²"}
        vetores = dict(zip(vZ,vPsi))
        gerar_arquivo_json('Psi'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.txt', vetores, titulos)
    elif choice == 3:
        print("Nothing here now.")
        time.sleep(1)
        raise SystemExit
    elif choice == 4:
        startall = timer()
        yy = 1.05**2
        vz=[]
        vCprzLA = []
        vCprzLAV = []
        vCprzE = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = np.arange(L*-0.5*10**3,(L*0.5*10**3)+step,step)
        print("Z's range:", step) #range de teste: 0.01
        print("Vector Z in mm", fstep, "Size:", len(fstep))
        #rar = float(input("r0 range: "))
        rar = 2*10**-13        
        frar = np.arange(-10*spot*10**-6,(10*spot*10**-6)+rar,rar)
        print("r0's range:", rar)
        print("Vector r0 in µm", frar, "Size:", len(frar))
        startfor = timer()
        for dz in fstep:
            aCprzE = CprzE(raiop,yy*raiop,0,0,dz*10**-3)
            print("Valor de CprzE para z = ", dz, "     CprzE[",raiop,",", yy*raiop ,", 0, 0,",dz,"*10**-3] = ", aCprzE)
            vCprzE.append(aCprzE)       
            aCprzLA = CprzLA(raiop,yy*raiop,0,0,dz*10**-3)
            print("Valor de CprzLA para z = ", dz, "     CprzLA[",raiop,",", yy*raiop ,",0,0,",dz,"*10**-3] = ", aCprzLA)
            vCprzLA.append(aCprzLA)   
            aCprzLAV = CprzLAV(raiop,yy*raiop,0,0,dz*10**-3)
            print("Valor de CprzLAV para z = ", dz, "     CprzLAV[",raiop,",", yy*raiop ,",0,0,",dz,"*10**-3] = ", aCprzLAV)
            vCprzLAV.append(aCprzLAV)             
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vz.append(dz)
        vr0=[]
        vCprxLA = []
        vCprxLAV = []
        vCprxE = []
        startfor = timer()
        for dr in frar:
            aCprxE = CprxE(raiop,yy*raiop,dr*10**-6,0,0)
            print("Valor de CprxE para r0 = ", dr, "     CprxE[",raiop,",", yy*raiop ,",",dr,"*10**-6, 0, 0] = ", aCprxE)
            vCprxE.append(aCprxE)       
            aCprxLA = CprxLA(raiop,yy*raiop,dr*10**-6,0,0)
            print("Valor de CprxLA para r0 = ", dr, "     CprxLA[",raiop,",", yy*raiop ,",",dr,"*10**-6, 0, 0] = ", aCprxLA)
            vCprxLA.append(aCprxLA)  
            aCprxLAV = CprxLAV(raiop,yy*raiop,dr*10**-6,0,0)
            print("Valor de CprxLAV para r0 = ", dr, "     CprxLAV[",raiop,",", yy*raiop ,",",dr,"*10**-6, 0, 0] = ", aCprxLAV)
            vCprxLAV.append(aCprxLAV)             
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vr0.append(dr)
        axes = plt.gca()
        axes.set_xlim([-10*spot*10**-6,(10*spot*10**-6)+rar])
        #ylim = [float(min(vBJx)),float(max(vBJx))]
        #axes.set_ylim(ylim)
        axes = plt.gca()
        axes.set_xlim([L*-0.5*10**3,(L*0.5*10**3)+step])
        #ylim = [float(min(vBJx)),float(max(vBJx))]
        #axes.set_ylim(ylim)
        plt.plot(vz,vCprzE, linestyle='-', color='black')
        plt.plot(vz,vCprzLA, linestyle='-', color='blue')
        plt.plot(vz,vCprzLAV, linestyle='--', color='red')
        plt.legend(['CprzE', 'CprzLA', 'CprzLAV'], loc='upper right')
        plt.grid(True)
        plt.xlabel('z (mm)')
        plt.ylabel('Cpr,z (m^2)')
        plt.show()
        plt.plot(vr0,vCprxE, linestyle='-', color='black')
        plt.plot(vr0,vCprxLA, linestyle='-', color='blue')
        plt.plot(vr0,vCprxLAV, linestyle='--', color='red')
        plt.legend(['CprxE', 'CprxLA', 'CprxLAV'], loc='upper right')
        plt.grid(True)
        plt.xlabel('r0 (µm)')
        plt.ylabel('Cpr,x (m^2)')
        plt.show()
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 5:
        ordem = 0
        r0 = 0
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]
        print("===============================================")
        print("order = 0 e r0 = 0")
        for na in naLista:
            print(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na

        ordem = 3
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        print("===============================================")
        print("ordem = 3 e r0 = 0")
        for na in naLista:
            print(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na
       
        ordem = 0
        r0 = 10*10**(-6)
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        print("===============================================")
        print("ordem = 0 e r0 = 10*10^(-6)")
        for na in naLista:
            print(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na, r0
        
        ordem = 3
        r0 = 10*10**(-6)
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        print("===============================================")
        print("ordem = 3 e r0 = 10*10^(-6)")
        for na in naLista:
            print(na, gnmTME(na,ordem + 1,r0,0,0), gnmTMLA(na,ordem + 1,r0,0,0), gnmTMLAV(na,ordem + 1,r0,0,0))
        del ordem, na, r0
        
        ordem = 2
        r0 = 10*10**(-6)
        z0 = 10**(-3)
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        print("===============================================")
        print("ordem = 2, r0 = 10*10^(-6) e z = 10^(-3)")
        for na in naLista:
            print(na, gnmTME(na,ordem + 1,r0,0,z0), gnmTMLA(na,ordem + 1,r0,0,z0), gnmTMLAV(na,ordem + 1,r0,0,z0))
 
        del na,r0,z0,ordem
    elif choice == 6:
        ordem = 0
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]
        print("===============================================")
        print("ordem = 0")
        for na in naLista:
            print(na, tau(na, ordem + 1, 0.2), fpi(na, ordem + 1, 0.2))
        
        ordem = 3
        naLista = [ordem + 1,ordem + 2,ordem + 5,ordem + 10,ordem + 25,ordem + 50.000000001,ordem + 100.000000001,ordem + 250.000000001,ordem + 500.000000001]     
        print("===============================================")
        print("ordem = 3")
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
        #step = 2.5
        startall = timer()
        ma = 1
        r0 = 0
        phi0 = 0
        z0 = 0
        vn = []
        vgnmTMLA = []
        vgnmTME = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = np.arange(1.001,400+step,step)
        print("n range:", step)
        print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTMLA = gnmTMLA(dna, ma, r0, phi0, z0)
            print("Valor de |gnmTMLA| para n = ", dna, "   |gnmTMLA[",dna,",1,0,0,0]| = ", abgnmTMLA)
            vgnmTMLA.append(float(abgnmTMLA.real))                  
            abgnmTME = gnmTME(dna, ma, r0, phi0, z0)
            print("Valor de |gnmTME| para n = ", dna, "   |gnmTME[",dna,",1,0,0,0]| = ", abgnmTME)
            vgnmTME.append(float(abgnmTME.real))            
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        axes = plt.gca()
        axes.set_xlim([0,400+step])
        ylim = [float(min(vgnmTMLA)*1.2),float(max(vgnmTMLA)*1.2)]
        axes.set_ylim(ylim)
        plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
        plt.legend(['gnmTME', 'gnmTMLA'], loc='upper right')
        plt.grid(True)
        plt.xlabel('n')
        plt.ylabel('gnmTM(n,±1,0,0,0)')
        a = plt.axes([.25, 0.65, .25, .25])
        testvgnmTME = [x*constants.kilo for x in vgnmTME]
        testvgnmTMLA = [x*constants.kilo for x in vgnmTMLA]
        plt.plot(vn,testvgnmTME, linestyle='-', color='black')
        plt.plot(vn,testvgnmTMLA, linestyle='--', color='red')
        plt.grid(True)
        plt.ylabel('(x10^-3)')
        plt.ylim(-4, 4)
        plt.xlim(300, 400)
        plt.show()
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
        titulos = {"n":"[gnmTME(n,±1,0,0,0),gnmTMLA(n,±1,0,0,0)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        titulos2 = {"n":"[gnmTME(n,±1,0,0,0),gnmTMLA(n,±1,0,0,0)]x10^-3"}
        vetores2 = dict(zip(vn,zip(testvgnmTME,testvgnmTMLA)))
        gerar_arquivo_json("teste1.txt", vetores, titulos)
        gerar_arquivo_json("teste2.txt", vetores2, titulos2)
    elif choice == 9:
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
        print("n range:", step)
        print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTMLA = abs(gnmTMLA(dna, ma, r0, phi0, z0))
            print("Valor de |gnmTMLA| para n = ", dna, "   |gnmTMLA[",dna,",2,0,0,0]| = ", abgnmTMLA)
            vgnmTMLA.append(float(abgnmTMLA.real))            
            abgnmTME = abs(gnmTME(dna, ma, r0, phi0, z0))
            print("Valor de |gnmTME| para n = ", dna, "   |gnmTME[",dna,",2,0,0,0]| = ", abgnmTME)
            vgnmTME.append(float(abgnmTME.real))            
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        axes = plt.gca()
        axes.set_xlim([-2,200*1.01+step])
        hectovgnmTME = [x*constants.hecto for x in vgnmTME]
        hectovgnmTMLA = [x*constants.hecto for x in vgnmTMLA]
        ylim = [float(min(hectovgnmTMLA)*-1),float(max(hectovgnmTMLA)*1.05)]
        axes.set_ylim(ylim)
        plt.plot(vn,hectovgnmTMLA, linestyle='-', color='black', linewidth=1.5)
        plt.plot(vn,hectovgnmTME, linestyle='--', color='red')
        plt.legend(['gnmTME', 'gnmTMLA'], loc='upper right')
        plt.grid(True)
        plt.xlabel('n')
        plt.ylabel('gnmTM(n,2,0,0,0)x(10^-2)')
        a = plt.axes([.5, 0.35, .25, .25])
        plt.plot(vn,hectovgnmTMLA, linestyle='-', color='black')
        plt.plot(vn,hectovgnmTME, linestyle='--', color='red')
        plt.grid(True)
        plt.ylabel('(x10^-2)')
        plt.ylim(0,0.08)
        plt.yticks(np.arange(0,0.08,0.02))
        plt.xlim(100, 200)
        plt.show()
        titulos = {"n":"[gnmTME(n,±1,0,0,0),gnmTMLA(n,±1,0,0,0)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json('gnm'+'_'+'fig2'+'_'+str(10*np.around(np.random.random_sample(), decimals=2))+'_'+str(date.today())+'_'+'r'+str(step)+'.txt', vetores, titulos)
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 10:
        #tipo 4
        #ordem 0
        #step 0.00000015
        startall = timer()
        na = 1
        ma = 1
        phi0 = 0
        r0 = 0
        vn = []
        vgnmTMLA = []
        vgnmTME = []
        com_vgnmTMLA = []
        com_vgnmTME = []
        #step = float(input("r0 range: "))
        step = float(tratar_entrada(sys.argv, "step"))
        fstep = np.arange(0,25*constants.micro+step,step)
        print("z0 range:", step)
        print("Vector z0", fstep, "Size:", len(fstep))
        startfor = timer()
        for dz in fstep:
            abgnmTMLA = gnmTMLA(na, ma, r0, phi0, dz)
            bbgnmTMLA = gnmTMLA(na, ma, r0, phi0, dz*constants.kilo)
            print("Valor de gnmTMLA para n = ", dz, " , ", abgnmTMLA)
            vgnmTMLA.append(float(abgnmTMLA.real))
            com_vgnmTMLA.append(float(bbgnmTMLA.real))       
            abgnmTME = gnmTME(na, ma, r0, phi0, dz)
            bbgnmTME = gnmTME(na, ma, r0, phi0, dz*constants.kilo)
            print("Valor de gnmTME para n = ", dz, " , ", abgnmTME)
            vgnmTME.append(float(abgnmTME.real))
            com_vgnmTME.append(float(bbgnmTME.real))                 
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dz)
        axes = plt.gca()
        axes.set_xlim([-0.5*constants.micro,20*constants.micro+0.5*constants.micro])
        ticks = axes.get_xticks()*10**6
        axes.set_xticklabels(ticks)
        ylim = [float(min(vgnmTMLA)*1.05),float(max(vgnmTMLA)*1.05)]
        axes.set_ylim(ylim)
        plt.plot(vn,vgnmTME, linestyle='-', color='black', linewidth=1.5)
        plt.plot(vn,vgnmTMLA, linestyle='--', color='red')
        plt.legend(['gnmTME', 'gnmTMLA'], loc='upper right')
        plt.grid(True)
        plt.xlabel('z0 (µm)')
        plt.ylabel('gnmTM(1,±1,0,r0,0)')
        titulos = {"n":"[gnmTM(1,±1,0,r0,0),gnmTM(1,±1,0,r0,0)]"}
        vetores = dict(zip(vn,zip(vgnmTME,vgnmTMLA)))
        gerar_arquivo_json("teste1.txt", vetores, titulos)
        #a = plt.axes([.15, 0.65, .25, .25])
        #testvgnmTMLAV = [x*constants.kilo for x in vgnmTMLAV]
        #plt.plot(vn,sympify(com_vgnmTMLA), linestyle='-', color='black')
        #plt.plot(vn,sympify(com_vgnmTME), linestyle='--', color='red')
        #plt.plot(vn,sympify(testvgnmTMLAV), linestyle='--', color='blue')
        #plt.grid(True)
        #plt.xlabel('z0 (mm)')
        #plt.axis('off')
        plt.show()
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 11:
        startall = timer()
        #tipo 1
        #ordem 0
        #step 0.0000003
        na = 200
        ma = 0
        phi0 = 0
        z0 = 0.3*10**-3
        vn = []
        vgnmTMLA = []
        #vgnmTMLAV = []
        vgnmTME = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = np.arange(0,50*constants.micro+step,step)
        print("r0 range:", step)
        print("Vector r0", fstep, "Size:", len(fstep))
        startfor = timer()
        for dr in fstep:
            abgnmTMLA = abs(gnmTMLA(na, ma, dr, phi0, z0))
            print("Valor de |gnmTMLA| para n = ", dr, "   |gnmTMLA[200,0,",dr,",0,0]| = ", abgnmTMLA)
            vgnmTMLA.append(abgnmTMLA)            
            abgnmTME = abs(gnmTME(na, ma, dr, phi0, z0))
            print("Valor de |gnmTME| para n = ", dr, "   |gnmTME[200,0,",dr,",0,0]| = ", abgnmTME)
            vgnmTME.append(abgnmTME)            
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dr)
        axes = plt.gca()
        axes.set_xlim([0,50*constants.micro])
        ticks = axes.get_xticks()*10**6
        axes.set_xticklabels(ticks)
        hectovgnmTME = [x for x in vgnmTME]
        hectovgnmTMLA = [x for x in vgnmTMLA]
        plt.plot(vn,sympify(hectovgnmTME), linestyle='-', color='black', linewidth=1.5)
        plt.plot(vn,sympify(hectovgnmTMLA), linestyle='--', color='red')
        plt.legend(['gnmTME', 'gnmTMLA'], loc='upper left')
        plt.grid(True)
        plt.xlabel('r0 (µm)')
        plt.ylabel('gnmTM(200,0,0,r0,0)')
        plt.show()
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 12:
        #tipo 1
        #ordem = 0
        #step = 2
        startall = timer()
        ma = 1
        r0 = 0
        phi0 = 0
        z0 = 0
        vn = []
        vgnmTME = []
        vgnmTMLA = []
        step = float(tratar_entrada(sys.argv,"step"))
        fstep = np.arange(1.001,400+step,step)
        print("n range:", step)
        print("Vector n", fstep, "Size:", len(fstep))
        startfor = timer()
        for dna in fstep:
            abgnmTMLA = gnmTMLA(dna, ma, r0, phi0, z0)
            print("Valor de |gnmTMLA| para n = ", dna, "   |gnmTMLA[",dna,",1,0,0,0]| = ", abgnmTMLA)
            vgnmTMLA.append(abgnmTMLA)                  
            abgnmTME = gnmTME(dna, ma, r0, phi0, z0)
            print("Valor de |gnmTME| para n = ", dna, "   |gnmTME[",dna,",1,0,0,0]| = ", abgnmTME)
            vgnmTME.append(abgnmTME)            
            endfor = timer()
            print("Repeatition Performance Time:", (endfor - startfor)/60, "minutes")
            vn.append(dna)
        axes = plt.gca()
        axes.set_xlim([0,400+step])
        ylim = [float(min(sympify(vgnmTMLA))*1.2),float(max(sympify(vgnmTMLA))*1.2)]
        axes.set_ylim(ylim)
        plt.plot(vn,sympify(vgnmTME), linestyle='-', color='black', linewidth=1.5)
        plt.plot(vn,sympify(vgnmTMLA), linestyle='--', color='red')
        plt.legend(['gnmTME', 'gnmTMLA'], loc='upper right')
        plt.grid(True)
        plt.xlabel('n')
        plt.ylabel('gnmTM(n,±1,0,0,0)')
        plt.show()
        endall = timer()
        print("Repeatition Performance Total Time:", (endall - startall)/60, "minutes")
    elif choice == 0:
        print("Ok, good bye.")
        time.sleep(1)
        raise SystemExit
    else:
        nope()
        raise SystemExit
