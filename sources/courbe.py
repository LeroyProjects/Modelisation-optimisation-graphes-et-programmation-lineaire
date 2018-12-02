#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 9:54:54 2017

@author: Cassandre et Sarah
"""

import matplotlib.pyplot as pyplot 

x = [0,	8,	25,	400,	481,	625,	675,	900,	924,	1054,	1140,	1400,	2025,	2500,	9801]
y1 = [0,	0.00875401496887207,	0.00875401496887207,	1.30344080924988,	0.686438798904419,	1.55001497268677,	1.77561497688293,	2.11529397964478,	4.34238219261169,	1.98075699806213,	3.09287190437317,	4.03554725646973,	6.14825701713562,	51.0498518943787,	82.3024868965149 ]
y2 = [0,	0.0357599258422852,	0.0262877941131592,	0.4444580078125,	0.239309072494507,	0.40519905090332,	0.310715913772583	,0.653588056564331,	0.783998966217041,	0.407552003860474	,0.656442880630493,	0.567651033401489,	1.11649894714355,	3.71912693977356,	7.50852990150452]
pyplot.scatter(x, y1, c = 'red')
pyplot.scatter(x, y2, c = 'blue')

x2 = [0,	8,	25,	400,	481,	625,	675,	900,	924,	1054,	1140,	1400]
plne = [0	,0.0385529994964599,	0.0360078811645508	,4.39851403236389,	1.18224310874939,	13.6439242362976	,18.0977387428284,	182.064480066299,	231.394896030426,	28.6933419704437,	272.40065407753	,53.3555529117584]

from numpy import *             #import de numpy
from pylab import *             #import de matplotlib.pylab
from scipy.stats import norm    #import du module norm de scipy.stats

c=[1,2,2]                       #Définition du polynome (t**2+2*t+2)
y=polyval(c,x)                  #Evaluation du polynome
yn=y+norm.rvs(size=len(x))      #Ajout d'un bruit gaussian
c_est = np.polyfit(x, y1, 2)     #regression polynomiale (degré 2)
y_est=polyval(c_est,x)
plot(x,y1,'o',label="Données Dynamiques", c = 'red')   #Affichage des points
plot(x,y_est,label="regression", c = 'red')#Affichage des points
title('Courbe programmation dynamique,méthode globale et PLNE')

c=[1,2,2]                       #Définition du polynome (t**2+2*t+2)
y=polyval(c,x)                  #Evaluation du polynome
yn=y+norm.rvs(size=len(x))      #Ajout d'un bruit gaussian
c_est = np.polyfit(x, y2, 2)     #regression polynomiale (degré 2)
y_est=polyval(c_est,x)
plot(x,y2,'o',label="Donnée Méthode_globale", c = 'blue')   #Affichage des points
plot(x,y_est,label="regression", c = 'blue')#Affichage des points


c=[1,2,2]                       #Définition du polynome (t**2+2*t+2)
y=polyval(c,x2)                  #Evaluation du polynome
yn=y+norm.rvs(size=len(x2))      #Ajout d'un bruit gaussian
c_est = np.polyfit(x2, plne, 2)     #regression polynomiale (degré 2)
y_est=polyval(c_est,x2)
plot(x2,plne,'o',label="Données PLNE", c = 'green')   #Affichage des points
plot(x2,y_est,label="regression", c = 'green')#Affichage des points

xlabel('Tailles grilles')
ylabel('Temps exécution en secondes')
legend()
show()                          #affichage des courbes









