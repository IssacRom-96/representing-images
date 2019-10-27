# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:53:41 2019

@author: Issac Romero
"""

import cv2 #IMPORTA LIBRERIA OPENCV
import matplotlib.pyplot as plt #IMPORTA LA LIBRERIA MATPLOT, ESTA SIMPLIFICADA COMO PLT

extractdir = r"C:/Users/Administrador/pedestrians128x64" #INSTANCIAMOS SET IMAGENES A UNA VARIABLE, EN ESTE CASO DATOS DE LA CARPETA PEDESTRIAN
plt.figure(figsize=(10, 6)) #CREA UNA NUEVA IMAGEN, CON UN TAMAÑO ESPECIFICO
##VISUALIZACION DE LAS IMAGENES##
for i in range(5): #BUCLE FINITO DE 5 DATOS
    filename = "%s/per0010%d.ppm" % (extractdir, i) #INSTACIAMOS DESDE DONDE EMPIEZA A LEER LAS IMAGENES NUESTRO CODIGO CON UNA VARIABLE
    img = cv2.imread(filename) #CARGAMOS LAS IMAGENES DEL ARCHIVO ESPECIFICADO, UNA POR UNA
    plt.subplot(1, 5, i + 1) #AGREGA UNA SUBTRAMA A LAS FIGURAS A MOSTRAR(FILAS, COLUMNAS, INDICE, ARGUMENTO)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #MUESTRA UNA IMAGEN, Y ESTE CONVIERTE UN IMAGEN DE UN ESPACIO DE COLOR A OTRO
    plt.axis('off') #METODO DE CONVENIENCIA PARA OBTENER O ESTABLECER ALGUNAS PROPIEDADES DEL EJE
    
win_size = (48, 96) #INDICAN EL TAMAÑO DE LA VENTANA A LEER DE LA IMAGEN EN LARGO Y ANCHO.
block_size = (16, 16) #INDICA EL TAMAÑO DE CADA  CUADRO
block_stride = (8, 8) #DISTANCIA ENTRE CELDAS
cell_size = (8, 8) #TAMAÑO DE LA CELDA
num_bins = 9 #PARA CADA UNO DE LAS CELDAS SE UTILIZAN 9 COMPARTIMIENTOS
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins) #IMPLEMENTACIÓN DEL DESCRIPTOR Y DETECTOR DE OBJETOS HOG (HISTOGRAMA DE GRADIENTES ORIENTADOS)

import numpy as np #IMPORTA LA LIBRERIA NUMPY
import random #IMPORTA LA LIBRERIA RANDON
#CODIGO DE ENTRENAMIENTO DE MUESTRAS POSITIVAS
random.seed(42) #MÓDULO ALEATORIO PARA INICIALIZAR EL GENERADOR DE NÚMEROS PSEUDOALEATORIOS PARA GENERAR LOS DATOS ALEATORIOS DETERMINISTAS, ESTE DE 42(PARAMETRO DE ALATEOREIDAD)
X_pos = [] #INTANCIAMOS UNA LISTA PARA EL CONJUNTO DE DATOS DE MUESTRAS POSITIVAS
for i in random.sample(range(1,900), 400): #SELECCION ALEATORIA DE DATOS DE NUESTRO DIRECTORIO, SELECCIONAMOS 400 DE LAS MAS DE 900
    filename = "%s/per%05d.ppm" % (extractdir, i) #INSTACIAMOS DESDE DONDE EMPIEZA A LEER LAS IMAGENES NUESTRO CODIGO CON UNA VARIABLE
    img = cv2.imread(filename) #CARGAMOS UNA IMAGEN DE ARCHIVO ESPECIFICADO, EN UNA VARIABLE 
    if img is None: #CONDICION SI NO HAY IMAGEN
        print('Could not find image %s' % filename) #INDICACION DE QUE NO HAY IMAGEN Y TE MUESTRA EL ARCHIVO 
        continue #Y CONTINUA EL COGIDO, NO SE QUEDA EN LA CONDICIONAL
    X_pos.append(hog.compute(img, (64, 64))) #AGREGA UN VALOR NUEVO A LA LISTA DE MUESTRAS POSITIVAS, ESTE DE LA REGION DE INTERES Y UN TAMAÑO #hog.compute
    
#EN ESTA PARTE DEL CODIGO SE MUESTRAN LOS DATOS DE ENTRENAMIENTO, EL TOTAL, Y EL VALOR DE LAS CARACTERISTICAS DE CADA MUESTRA POSITIVA
X_pos = np.array(X_pos, dtype=np.float32) #INTANCIAMOS UNA MATRIZ CON LOS DATOS DE LA LISTA DE MUESTRAS POSITIVAS
y_pos = np.ones(X_pos.shape[0], dtype=np.int32) #INTANCIAMOS UN CONJUNTO DE FORMAS RELLENOS CON UNOS
X_pos.shape, y_pos.shape #MOSTRAMOS EL VALOR DE LOS DOS CONJUNTOS DE DATOS

datadir = r"C:\Users\Administrador\paquete_de_imagenes" #INTANCIAMOS UNA CARPETA DE DATOS EN UNA VARIABLE
negset = "pedestrians_neg" #INTANCIAMOS UNA CARPETA CON UN SET DE IMAGENES DE LUGARES
negdir = "%s/%s" % (datadir, negset) #INTANCIAMOS DOS VARIABLE EN UNA, ESTAS DE LOS DOS DATOS ANTERIORES
#CODIGO DE ENTRENAMINTO DE MUESTRAS NEGATIVAS
import os #IMPORTAMOS EL MODULO OS, ESTE PERMITE ACCEDER A FUNCIONES DEPENDIENTES DEL SISTEMA OPERATIVO
hroi = 128 #REGION DE INTERES PARA LA ULTURA DE IMAGEN
wroi = 64 #REGION DE INTERES PARA LA ANCHURA DE IMAGEN
X_neg = [] #INTANCIAMOS UNA LISTA PARA EL CONJUNTO DE DATOS DE MUESTRAS NEGATIVAS
for negfile in os.listdir(negdir): 
    filename = '%s/%s' % (negdir, negfile) #INTANCIAMOS EN UNA VARIABLE DOS DATOS
    img = cv2.imread(filename) #CARGAMOS UNA IMAGEN DE ARCHIVO ESPECIFICADO, EN UNA VARIABLE
    img = cv2.resize(img, (512, 512)) #READIMENSIONAMOS EL TAMAÑO DE LA IMAGEN ANTERIOR
    for j in range(5): #BUCLE FINITO DE 5 DATOS
        #ELIGIMOS ALEATORIAMENTE LA COORDENADA DE LA ESQUINA SUPERIOR IZQUIERDA
        rand_y = random.randint(0, img.shape[0] - hroi)
        rand_x = random.randint(0, img.shape[1] - wroi)
        roi = img[rand_y:rand_y + hroi, rand_x:rand_x + wroi, :]
        X_neg.append(hog.compute(roi, (64, 64))) #AGREGA UN VALOR NUEVO A LA LISTA DE MUESTRAS NEGATIVAS, ESTE DE LA REGION DE INTERES Y UN TAMAÑO
        
#EN ESTA PARTE DEL CODIGO SE MUESTRAN LOS DATOS DE ENTRENAMIENTO, EL TOTAL, Y EL VALOR DE LAS CARACTERISTICAS DE CADA MUESTRA NEGATIVAS
X_neg = np.array(X_neg, dtype=np.float32) #INTANCIAMOS UNA MATRIZ CON LOS DATOS DE LA LISTA DE MUESTRAS NEGATIVAS
y_neg = -np.ones(X_neg.shape[0], dtype=np.int32) #INTANCIAMOS UN CONJUNTO DE FORMAS RELLENOS CON UNOS NEGATIVOS
X_neg.shape, y_neg.shape #MOSTRAMOS EL VALOR DE LOS DOS CONJUNTOS DE DATOS

X = np.concatenate((X_pos, X_neg)) #INTANCIAMOS UNA VARIABLE COMO EJE X, ESTE UNE LAS DOS MATRICES, POSITIVA Y NEGATIVA 
y = np.concatenate((y_pos, y_neg)) #INTANCIAMOS UNA VARIABLE COMO EJE Y, ESTE UNE LAS DOS MATRICES, POSITIVA Y NEGATIVA 

from sklearn import model_selection as ms #DE SKLEARN IMPORTAMOS MODEL_SELECCION
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)#DIVIDIR MATRIVES O MATRICES EN TRENES ALEATORIOS Y SUBCONJUNTO DE PRUEBAS(PARAMETROS DE MATRICES,PROPORCION DEL CONJUNTO DE DATOS, SEMILLA UTILIZADA PARA LA GENERACION DE DATOS)

def train_svm(X_train, y_train): #DEFINIMOS LA FUNCION TRAIN_SVM, CON DOS ATRIBUTOS
    svm = cv2.ml.SVM_create() #INSTANCIAMOS EN UNA VARIABLE(SVM) ESTE CREA UN MODELO VACÍO 
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm #NOS REGRESA EL VALOR DE SVM

def score_svm(svm, X, y): #DEFINIMOS UNA FUNCION SCORE_SVM,CON TRES ATRIBUTOS
    from sklearn import metrics #DE SKLEARN IMPORTAMOS METRICAS
    _, y_pred = svm.predict(X) #REALIZAR LA CLASIFICACIÓN DE MUESTRAS EN X Y ESTA LA INTANCIAMOS
    return metrics.accuracy_score(y, y_pred) #NO REGRESA EL CALCULO DE LA PRESICION DE LOS SUBCONJUNTOS 

svm = train_svm(X_train, y_train)

score_svm(svm, X_train, y_train)#DEVUELVE LA PRECISIÓN MEDIA EN LOS DATOS Y ETIQUETAS DE PRUEBA DADOS

score_svm(svm, X_test, y_test)#DEVUELVE LA PRECISIÓN MEDIA EN LOS DATOS Y ETIQUETAS DE PRUEBA DADOS

score_train = []  #DEFINIMOS LA LISTA SCORE_TRAIN 
score_test = []   #DEFINIMOS LA LISTA SCORE_TEST
for j in range(3):   #CICLO FOR PARA CONTAR HASTA 3
    svm = train_svm(X_train, y_train)     #DE LA BIBLIOTECA SVM DENTRO DE SKLEARN
    score_train.append(score_svm(svm, X_train, y_train))    #SE AGREGAN LOS DATOS A LA LISTA SCORE_TRAIN
    score_test.append(score_svm(svm, X_test, y_test))       #SE AGREGAN LOS DATOS A LA LISTA SCORE_TEST
    
    _, y_pred = svm.predict(X_test)               #PREDICT ES UNA FUNCION DENTRO DE LA BIBLIOTECA DE SVM, UTILIZADO PARA HACER PREDICCIONES
    false_pos = np.logical_and((y_test.ravel() == -1), (y_pred.ravel() == 1))  #DE LA LIBRERIA DE NUMPY PARA DEVOLVER UNA MATRIZ APLANADA CONTIGUA
    if not np.any(false_pos):     #TOMA DATOS ITERABLES (LISTA,CADENA, DISCCIONARIO), SI NO TIENE ALGUN ITERABLE 
        print('done')             #IMPRIME "DONE"
        break
    X_train = np.concatenate((X_train, X_test[false_pos, :]), axis=0)   #FUSIONAR O COMBINAR CADENAS
    y_train = np.concatenate((y_train, y_test[false_pos]), axis=0)      #""          ""
    
score_train  #IMPRIME LA LISTA
score_test   #IMPRIME LA LISTA 

img_test = cv2.imread(r'C:\Users\Administrador\paquete_de_imagenes\IMG_20180916_142552574.jpg')  #DECLARAMOS LA VARIABLE CON UNA DIRECCION DE UNA IMAGEN

stride = 16       #VARIABLE DECLARADA
found = []        #CREACION DE LA LISTA LLAMADA FOUND[]
for ystart in np.arange(0, img_test.shape[0], stride):         #LA FUNCION ARANGE DEVUELVE LOS VALORES ESPACIADOS UNIFORMEMENTE DENTRO DE UN INTERVAlO DADO
    for xstart in np.arange(0, img_test.shape[1], stride):     #MISMA FUNCION .ARANGE
        if ystart + hroi > img_test.shape[0]:                  #LA FUNCION .SHAPE DEVUELVE LAS DIMENSIONES DE LA MATRIZ             
            continue
        if xstart + wroi > img_test.shape[1]:
            continue
        roi = img_test[ystart:ystart + hroi, xstart:xstart + wroi, :]  #SE DEFINE UNA VARIABLE 
        feat = np.array([hog.compute(roi, (64, 64))])     #TODOS LOS ELEMENTOS DEL ARRAY DEBEN SER DEL MISMO TIPO 
        _, ypred = svm.predict(feat)                      #FUNCION PARA LA PREDICCION DE DATOS  
        if np.allclose(ypred, 1):                         #SE UTILIZA ESTA FUNCION PARA ENCONTRAR SI DOS MATRICES SON IGUALES EN CUESTION DE ELEMENTOS 
            found.append((ystart, xstart, hroi, wroi))    #SE AGREGAN VALORES A LA LISTA 
            
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

rho, _, _ = svm.getDecisionFunction(0)
sv = svm.getSupportVectors()
hog.setSVMDetector(np.append(sv[0, :].ravel(), rho))

hogdef = cv2.HOGDescriptor()

hogdef.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

found, _ = hogdef.detectMultiScale(img_test)

fig = plt.figure(figsize=(10, 6))          #AQUI DEFINIMOS EL TAMAÑO DE LA IMAGEN, DIMENSIONES 
ax = fig.add_subplot(111)                  #FUNCION PARA GRAFICAR PARA AGREGAR LA VARIABLE FIG
ax.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))    #CONFIGURAMOS LOS TONOS DE COLOR DENTRO DE LA IMAGEN 
from matplotlib import patches             #IMPORTAMOS LA FUNCION PATCHES DENTRO DE LA LIBRERIA DE MATPLOTLIB
for f in found:                            #CICLO FOR DENTRO DE LOS VALORES DE FOUND 
    ax.add_patch(patches.Rectangle((f[0], f[1]), f[2], f[3], color='r', linewidth=3, fill=False)) #FUNCION PARA IDENTIFICAR LOS DATOS YA ENTRENADOS Y DETECTAR ERSONAS DENTRO DE UNA IMAGEN 
plt.savefig('detected.png')                #GUARDA LA IMAGEN YA PROCESADA PAERA IDENTIFICAR PERSONAS.
    
    
    
    
    
    
    