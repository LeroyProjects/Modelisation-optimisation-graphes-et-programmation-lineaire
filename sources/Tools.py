#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:23:48 2017

@author: cassandre
"""

import pylab as plt
from matplotlib.collections import PatchCollection
from  matplotlib.patches import  Rectangle

# Classe Outil :
class Tools :
    def parse_ligne(self, ligne):
        return list(map(int, ligne.split()))


    def parse_instance(self, file_path):    
        lignes, colonnes = [], []
        with open(file_path) as file:
            for line in file:
                if line != "#\n":
                    lignes.append(self.parse_ligne(line))
                else:
                    break
            for line in file:
                colonnes.append(self.parse_ligne(line))
        return lignes, colonnes
    
    def affichage(self, grille, titre):
        n = len(grille)
        m = len(grille[0])
        plt.rcParams['figure.figsize'] = int((m*5)/ n) , 5
        plt.subplots_adjust(bottom=0.02, top=0.92, left=0.02, right=0.98)
        graphe = plt.subplot(1, 1, 1)
        graphe.get_xaxis().set_visible(False)
        graphe.get_yaxis().set_visible(False)
        plt.axis([0, len(grille[0]), n , 0])
        plt.suptitle(titre, fontsize=24)
    
        CN = []
        CB = []
        CG = []
        for i in range(n):
            for j in range(len(grille[i])):
                if grille[i][j] == 1:
                    CN.append(Rectangle((j, i), 1, 1, color="black"))
                elif grille[i][j] == -1:
                    CB.append(Rectangle((j, i), 1, 1, color="white"))
                elif grille[i][j] == 0:
                    CG.append(Rectangle((j, i), 1, 1, color="grey"))
        if CN != []:
            graphe.add_collection(PatchCollection(CN, match_original=True))
        if CB != []:
            graphe.add_collection(PatchCollection(CB, match_original=True))
        if CG != []:
            graphe.add_collection(PatchCollection(CG, match_original=True))
        plt.show()