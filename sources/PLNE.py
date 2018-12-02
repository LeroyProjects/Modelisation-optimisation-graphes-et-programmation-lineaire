#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:07:43 2017

@author: Sarah et Cassandre

"""  

from gurobipy import *
import numpy as np
import time as time
from Tools import Tools
"""
import os
os.chdir("/Users/cassandre/Desktop/partiel")
"""

import os
os.chdir("/Users/cassandre/Desktop/Projet_MOGPL_Lachiheb_Leroy/instances")
def hors_intervalle_exlus(nbLignes, nbColonnes, sequence):
    intervalles_possible = []
    # Iteration sur les lignes :
    for i in range(nbLignes) :
        # Récupere la sequence de la ligne i :
        sequence_i = sequence[i]
        if len(sequence_i) == 0 :
            intervalles_possible.append([])
        else :
            # Ietration sur les colonnes :
            for cursor in range (nbColonnes) :
                curseur = cursor
                # Iteration sur chaque bloc t de la sequence_i:
                valide = True
                for bloc_t in range(len(sequence_i)) :
                    # Si cursor ne sort pas de la grille :
                    if curseur + sequence_i[bloc_t]  <= nbColonnes - 1:
                        curseur += sequence_i[bloc_t] + 1 
                    elif curseur + sequence_i[bloc_t] -1 == nbColonnes - 1 :
                        curseur += sequence_i[bloc_t] -1
                    else :
                        valide = False
                        break
                if valide :
                    # Insertion dans la liste des intervalles possibles :
                    if cursor == 0 :
                        intervalle_i = [[]for t in range (len(sequence_i))]
                    for t in range (len(sequence_i)) :
                        intervalle_i[t].append(cursor)
                        cursor += sequence_i[t] + 1
                else :
                    break
            # Insertion les intervalles possibles dans la liste final :
            intervalles_possible.append(intervalle_i)
    return intervalles_possible

def resolution_grille_model(x, grille, N, M) :
    # Resolution de la grille :
    for i in range(N):
        for j in range(M):
            if grille[i,j] == 0:
                # Recupere la valeur Binaire de Xi,j apres optimisation:
                xi_j_opt =  x[i,j].x
                # Si la xi_j_opt=0 alors on le remplace dans la grille par -1 (=Blanc) et xi_j_opt=1 par 1(=Noir) :
                if xi_j_opt == 0:
                    grille[i,j] = -1
                else:
                    grille[i,j] = 1   
    return grille

def PLNE(grille, contraintes_lignes, contraintes_colonnes):
    # Declaration des tailles lignes et colonnes : 
    N , M  = len(contraintes_lignes), len(contraintes_colonnes)
    
    # Declaration d'un model programme lineaire :
    m  = Model("Tomographie")
    
    # Reduction des variabes Y_ijt en exluant les variables qui ne sont pas possible :
    Y_possibles = hors_intervalle_exlus(N, M, contraintes_lignes)
    # Reduction des variables Z jit en excluant les variables qui ne sont pas possible :
    Z_possibles = hors_intervalle_exlus(M, N, contraintes_colonnes)
    
    
    # Declaration des variables de decision X i,j :
    x = np.array([[m.addVar(vtype=GRB.BINARY, name="x,%d,%d" % (i,j)) for j in range(M)]for i in range(N)])

    # Declaration variable Y i,j,t :
    y = np.zeros((N, M, len(contraintes_lignes)), dtype=object)
    for i in range (N) :
        for j in range (M) :
            for t in range (len(contraintes_lignes[i])) :
                if j in Y_possibles[i][t] :
                    y[i,j,t] = m.addVar(vtype=GRB.BINARY, name="y %d,%d,%d" % (i,j,t))
                else :
                    y [i,j,t] = None

    # Declaration variable Z j,i,t :
    z = np.zeros((M, N, len(contraintes_colonnes)), dtype=object)
    for j in range (M) :
        for i in range (N) :
            for t in range (len(contraintes_colonnes[j])) :
                if i in Z_possibles[j][t] :
                    z[j,i,t] = m.addVar(vtype=GRB.BINARY, name="y %d,%d,%d" % (j,i,t))
                else :
                    z[j,i,t] = None
    
    # maj du modele pour integrer les nouvelles variables :
    m.update()
    
    # Declaration d'une expression lineaire avec Xi,j :
    expr = LinExpr()
    # Somme de toute les contraintes xij :
    expr = quicksum(sum(x))
    
    # Affichage de l'expression lineaire :
    #print("\n",expr,"\n")
    
    # Declaration des containtes :
   
    # Declaration des contraintes Y i,j,t pour les lignes:
    for i in range(N):
        for t in range(len(contraintes_lignes[i])):
            # Recupere l'ensemble des contraintes pour chaque lignes :
            contrainte_Yi_j_t = [y[i,k][t] for k in range(M) if y[i,k][t] != None]
            # Si il existe au moins une contrainte :
            if contrainte_Yi_j_t:
                # La somme des contraintes etant egale a 1, montre qu'il y a une case noir representant le debut d'un bloc dans la ligne:
                m.addConstr(quicksum(contrainte_Y for contrainte_Y in contrainte_Yi_j_t) == 1)
     

    # Declaration des contraintes Z i,j,t pour les colonnes:       
    for j in range(M):
        for t in range(len(contraintes_colonnes[j]) ):
            # Recupere l'ensemble des contraintes pour chaque colonnes :
            contrainte_Zi_j_t = [z[j,i][t] for i in range(N) if z[j,i][t] != None]
            # Si il existe au moins une contrainte :
            if contrainte_Zi_j_t :
                 # La somme des contraintes etant egale a 1, montre qu'il y a une case noir representant le debut d'un bloc dans la colonne:               
                m.addConstr(quicksum(contrainte_Z for contrainte_Z in contrainte_Zi_j_t) == 1)
                
               
    # Declaration des constraintes d'intervalle_noir et Decalage:                  
    for i in range(N):  #ligne
        for j in range(M): #colonne
            # Contraintes_lignes :
            for t in range(len(contraintes_lignes[i])): # on itere sur le nombre de contrainte dans une ligne i:
                if y[i,j][t] != None:
                    # Recupere dans l'intervalle de la case (i,j) à (i,j+st-1) les xij  :
                    liste_xij = [x[i,k] for k in range(j,j+contraintes_lignes[i][t]) if k < M ]
                    if liste_xij:
                        m.addConstr(y[i,j][t]*contraintes_lignes[i][t] <= quicksum(xij for xij in liste_xij))
                        # Impossible complexite peut devenir quadratique :
                        #=>m.addConstr((y[i,j][t]*contraintes_lignes[i][t] - quicksum(xij for xij in liste_xij))) <= 0) 
                    
                    # Contrainte sur le decalage :
                    for t_i in range(t+1 ,len(contraintes_lignes[i])):
                        # Requere les z_jk compris dans l'intervalle (i,j) a (i,j+st) pour verifier qu'il y est qu'un bloc qui commence.
                        liste_yij = [y[i,k][t_i] for k in range(j+contraintes_lignes[i][t]+1) if y[i,k][t_i] != None and k < M]
                        if liste_yij:
                            m.addConstr((y[i,j][t] + quicksum(one_bloc for one_bloc in liste_yij))<= 1)
                            #m.addConstr((len(liste_yij)*y[i,j][t] + quicksum(one_bloc for one_bloc in liste_yij))<= len(liste_yij))

    
            # Contraintes_colonnes :
            for t in range(len(contraintes_colonnes[j])): # on itere sur le nombre de contrainte dans une colonne j:
                if z[j,i][t] != None:
                    # Recupere dans l'intervalle de la case (j,i) à (i,i+st-1) les xji :              
                    liste_xji = [x[k,j] for k in range(i,i+contraintes_colonnes[j][t]) if k < N ]
                    if liste_xji:
                        m.addConstr(z[j,i][t]*contraintes_colonnes[j][t] <= quicksum(xji for xji in liste_xji))
                        # Impossible complexite peut devenir quadratique :
                        #=>m.addConstr(z[j,i][t]*(contraintes_colonnes[j][t] - quicksum(xji for xji in liste_xji))) <= 0) 
                    
                    # Contrainte sur le decalage :
                    for t_i in range(t+1,len(contraintes_colonnes[j])):
                        # Requere les z_jk compris dans l'intervalle (j,i) a (j,i+st) pour verifier qu'il y est qu'un bloc qui commence.
                        liste_zji = [z[j,k][t_i]  for k in range(i+contraintes_colonnes[j][t]+1) if z[j,k][t_i] != None and k < N]
                        if liste_zji:
                            m.addConstr((z[j,i][t] + quicksum(one_bloc for one_bloc in liste_zji)) <= 1)
                   
    # Definition de l'objectif :
    m.setObjective(expr,GRB.MINIMIZE)
    
    # Resolution :
    m.optimize()   
        
    # Resolution de la grille :
    resolution_grille_model(x, grille, N, M)
                
    # Retourne la grille resolue par PLNE :
    return grille

tools = Tools()
liste = [11]
for i, instance in enumerate(liste):
    # instancie le nom de chaque fichier :
    nom_fichier = str(instance)+".txt"
    # Recupere les contraintes lignes et colonnes dans un fichier : 
    contraintes_lignes, contraintes_colonnes = tools.parse_instance(nom_fichier)
    # Initialisation de la grille :
    grille = np.zeros((len(contraintes_lignes),len( contraintes_colonnes)))
    # Appel du programme PLNE :
    t1 = time.time()
    grille = PLNE(grille, contraintes_lignes, contraintes_colonnes)
    t2 = time.time()
    # Affichage de la grille :
    tools.affichage(grille,"instances : "+str(instance))
    print (t2-t1)


