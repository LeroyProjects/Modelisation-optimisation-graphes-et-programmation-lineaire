#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 17:07:43 2017

@author: Sarah et Cassadnre

"""  
from gurobipy import *
import numpy as np
import time as time
from Tools import Tools

# Declaration de variable globale:
CASE_WHITE = -1
CASE_EMPTY = 0
CASE_BLACK = 1

import os
os.chdir("/Users/cassandre/Desktop/partiel")

""" PRATIE PROGRAMMATION DYNAMIQUE :"""

class GrilleNonResolvable(Exception):
    """Leve une Exception en cas de Grille non resolvable"""
    def __init__(self, message):
        Exception.__init__(self, message)

def Top_Down_generalization(ligne, sequence, memoire_cache):
    j, l = len(ligne), len(sequence)    
    # On regarde si la clef (j,l) a deja ete calcule:
    if (j,l) in memoire_cache :
        return memoire_cache[(j,l)]
    # Si la clef n'a pas ete calcule ulterieurement:
    if l == 0:
        memoire_cache[(j,l)] = np.all(ligne <= 0)
    elif j < sequence[-1]:
        memoire_cache[(j,l)] =  False
    elif l==1 and j == sequence[-1]:
        memoire_cache[(j,l)] =  np.all(ligne >= 0)
    elif l > 1 and j == sequence[-1]:
        memoire_cache[(j,l)] =  False
    elif j > sequence[-1]:
        if ligne[j - 1] == CASE_WHITE:
            memoire_cache[(j,l)] =  Top_Down_generalization(ligne[:j - 1],sequence,memoire_cache)
        else :
            if ligne[j-1] == CASE_EMPTY and (j-1,l) in memoire_cache :
                memoire_cache[(j,l)] = memoire_cache[(j-1,l)]
            else :
                # Decomposition de la condition en plusieur variable :
                insert_bloc = np.all(ligne[j-sequence[-1]:] >= 0)
                one_bloc = l==1 and np.all(ligne[:j-sequence[-1]] <=0)
                several_bloc = (ligne[j-sequence[-1]-1] <= 0) and \
                            Top_Down_generalization(ligne[:j - sequence[-1] - 1],sequence[:-1],memoire_cache)
                # Condition final pour validation :
                condition = insert_bloc and (one_bloc or several_bloc)
                if ligne[j - 1] == CASE_BLACK:
                    memoire_cache[(j,l)] = condition
                elif ligne[j - 1] == CASE_EMPTY:
                    memoire_cache[(j,l)] =  condition or Top_Down_generalization(ligne[:j - 1],sequence,memoire_cache)
    return memoire_cache[(j,l)]

def Top_Down_generalizationInit(ligne, sequence):
    return Top_Down_generalization(ligne, sequence,{})

def Test_dynamic(grille,sequences_lignes, sequences_colonnes, indiceAVoir) :
    indiceAajouter = set()
    for i in indiceAVoir:
        # Récupere seulement les case à qui sont non coloriée :
        # la fonction where, nous donne l'indice de ces cases non coloriées :
        for j in np.where(grille[i] == CASE_EMPTY)[0]:
            # Recupere les sequences d'une ligne et une colonne pour la case (i,j)
            lig = sequences_lignes[i]
            col = sequences_colonnes[j]
            # Test avec une case (i,j) BLACK :
            grille[i][j] = CASE_BLACK
            test_black = Top_Down_generalizationInit(grille[i],lig) and Top_Down_generalizationInit(np.transpose(grille)[j], col)
            #test_black = Bottom_Up_generalization(grille[i],lig) and Bottom_Up_generalization(np.transpose(grille)[j], col)
            # Test avec une case (i,j) White :
            grille[i][j] = CASE_WHITE
            test_white =  Top_Down_generalizationInit(grille[i],lig) and Top_Down_generalizationInit(np.transpose(grille)[j], col)
            #test_white =  Bottom_Up_generalization(grille[i],lig) and Bottom_Up_generalization(np.transpose(grille)[j], col)
            if not test_black and not test_white:
                # Leve une exception en cas de grille non resolvable:
                # Cette exception est rattraper dans le main
                raise GrilleNonResolvable("Grille non resolvable pour la ligne "+i+",la colonne "+j)
            elif test_black and test_white:
                # Impossible de decider de la couleur de la case :
                grille[i][j] = CASE_EMPTY
                #indiceAajouter.add(j) # Permet au chien de s'afficher en entier
            elif test_black and not test_white:
                grille[i][j] = CASE_BLACK
                indiceAajouter.add(j) 
            else:
                grille[i][j] = CASE_WHITE
                indiceAajouter.add(j) 
    return indiceAajouter              

def propagation (sequences_lignes, sequences_colonnes) :
    # Récupere la taille des contraintes sur les lignes et colonnes : 
    N, M = len(sequences_lignes), len(sequences_colonnes)
    # Initialisation de la grille avec une valeur de type CASE_EMPTY = 0 :
    grille = np.full((N, M), CASE_EMPTY)
    # Initialisation des lignes et colonnes a voir :
    # On choisit de prendre la fonction set() afin de ne pas avoir de doublons 
    lignesAVoir, colonnesAVoir = set(range(N)), set()
    # Debut d'Algorithme :
    while lignesAVoir or colonnesAVoir:
        # Test toute les lignes et retourne les colonnes a voir,
        #c'est à dire les colonnes ou les case (i,j) d'une ligne ont été modifié:
        colonnesAVoir = Test_dynamic(grille, sequences_lignes, sequences_colonnes,lignesAVoir)
        # Les lignes étant toute parcourut, on met l'ensemble à vide
        lignesAVoir = set()
        # Test toute les colonnes:
        #c'est à dire les lignes ou les case (i,j) d'une colonnes ont été modifié:
        lignesAVoir = Test_dynamic(np.transpose(grille), sequences_colonnes, sequences_lignes, colonnesAVoir)
        colonnesAVoir = set()
    return grille

""" PARTIE PROGRAMMATION LINAIRE """

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
    
    # Declaration des containtes :--------------------------------------------------->
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
    
    """ Contrainte Ajouter pour la Methode globale """
    # Insertion des contraintes via la propagation :
    # Chaque case (i, j) décidée lors de la phase de propagation on lui associe une valeur
    # 1 si la case est Noir et 0 si la case est blanche :
    for i in range(N):
        for j in range(M):
            # Si la case est (i,j) est White :
            if grille[i,j] == CASE_WHITE:
                m.addConstr(x[i,j] == 0)
                # Si la case est blanche on sait que ce n'est pas le commencement d'un bloc t
                # Parcours la contrainte de la ligne associee à la case (i,j) :
                for t in range(len(contraintes_lignes[i])):
                    # Si la contrainte Yi,j,t existe :
                    if y[i,j][t] != None:
                            m.addConstr(y[i,j][t] == 0)
                # Parcours la contrainte de la colonne associee à la case (i,j) : 
                for t in range(len(contraintes_colonnes[j])):
                    # Si la contrainte Zj,i,t existe :
                    if z[j,i][t] != None:
                        m.addConstr(z[j,i][t] == 0)
            # Si la case est (i,j) est Black :
            if grille[i,j] == CASE_BLACK:
                m.addConstr(x[i,j] == 1) 
                
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
liste = [i for i in range (0,1)]
for i, instance in enumerate(liste):
    # instancie le nom de chaque fichier :
    nom_fichier = "oiseau.txt"#str(instance)+".txt"
    # Recupere les contraintes lignes et colonnes dans un fichier : 
    contraintes_lignes, contraintes_colonnes = tools.parse_instance(nom_fichier)
    # Propagation : 
    try :
        grille = propagation(contraintes_lignes, contraintes_colonnes)
    except GrilleNonResolvable :
        print("Exception : Grille non Resolvable")
    # Appel du programme PLNE :
    t1 = time.time()
    grille = PLNE(grille, contraintes_lignes, contraintes_colonnes)
    t2 = time.time()
    # Affichage de la grille :
    tools.affichage(grille,"instances : "+str(instance))
    print (t2-t1)


