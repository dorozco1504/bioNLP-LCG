###########################################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# AUTOR: Carlos Mendez, Cristian Jesús González 
# VERSION: 1.0
# DESCRIPCIÓN: Programa que realiza las pruebas para la evaluación de modelos clasificadores y calcula su valor F, exhaustividad y precisión    
# USO : Clasificación de archivos de texto haciendo uso de modelos SVM previamente entrenados.
# REQUISITOS : Archivos de entrada de texto plano, librerías mencionadas y sus dependencias y recibe parámetros enlistados
# FORMATO DE ENTRADA: Texto plano, recibe el conjunto de datos de prueba, el modelo clasificador a probar y parámetros. 
# FORMATO DE SALIDA : Texto plano con oraciones clasificadas, puntuación de parámetros valor F, exhaustividad y precisión
# LANGUAJE : Python
###########################################################################################################################

# -*- encoding: utf-8 -*-

import os
from time import time
from optparse import OptionParser
from nltk import word_tokenize
import sys
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from sklearn.svm import SVC
import scipy.stats
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection

__author__ = 'CMendezC'

# Objetivo: classify text files by using trained SVM model and vectorizer.
#   Model and vectorizer must exist

# Parameters
#   1) --inputPath Path to read TXT files to classify.
#   2) --inputFile File to read text to classify (one per line).
#   3) --outputPath Path to place classified TXT files.
#   4) --modelPath Parent path to read trained model and vectorizer.
#   5) --modelName Name of model and vectorizer to load.


#   11) --clasePos Clase positiva para clasificación
#   12) --claseNeg Clase negativa para clasificación

# Ouput:
#   1) A file with classified sentences (one per line), with class.

# Execution:

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path to read file with features extracted to classify", metavar="PATH")
    parser.add_option("--inputFile", dest="inputFile",
                      help="File to read text to classify (one per line)", metavar="FILE")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Path to place classified text", metavar="PATH")
    parser.add_option("--outputFile", dest="outputFile",
                      help="Output file name to write classified text", metavar="FILE")
    parser.add_option("--modelPath", dest="modelPath",
                      help="Path to read trained model", metavar="PATH")
    parser.add_option("--modelName", dest="modelName",
                      help="Name of model and vectorizer to load", metavar="NAME")
    # Clase positiva para clasificación
    parser.add_option("--clasePos", dest="clasePos",
                      help="Clase positiva del corpus", metavar="CLAS")
    # Clase negativa para clasificación
    parser.add_option("--claseNeg", dest="claseNeg",
                      help="Clase negativa del corpus", metavar="CLAS")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read file with features extracted to classify: " + str(options.inputPath))
    print("File to read text to classify (one per line): " + str(options.inputFile))
    print("Path to place classified TXT file: " + str(options.outputPath))
    print("Output file name to write classified text: " + str(options.outputFile))
    print("Path to read trained model, vectorizer, and dimensionality reduction: " + str(options.modelPath))
    print("Name of model, vectorizer, and dimensionality reduction to load: " + str(options.modelName))
    print("Positive class: " + str(options.claseNeg))
    print("Negative class: " + str(options.clasePos))

    t0 = time()

    listSentences = []

    with open(os.path.join(options.inputPath, options.inputFile), 'r', encoding='utf8', errors='replace') as iFile:
        for line in iFile.readlines():
            line = line.strip('\n')
            listSentences.append(line)

    print("Classifying texts...")
    print("   Loading model and vectorizer: " + options.modelName)
    if options.modelName.find('.SVM.'):
        classifier = SVC()
    classifier = joblib.load(os.path.join(options.modelPath, 'models', options.modelName + '.mod'), mmap_mode=None)
    vectorizer = joblib.load(os.path.join(options.modelPath, 'vectorizers', options.modelName + '.vec'))

    matrix = csr_matrix(vectorizer.transform(listSentences), dtype='double')
    print("   matrix.shape " + str(matrix.shape))

    # Classify corpus list
    y_predicted = classifier.predict(matrix)

    print("   Predicted class list length: " + str(len(y_predicted)))

    with open(os.path.join(options.outputPath, options.outputFile), "w", encoding="utf-8") as oFile:
        oFile.write('SENTENCE\tPREDICTED_CLASS\n')
        for s, pc in zip(listSentences, y_predicted):
            oFile.write(s + '\t' + pc + '\n')

    print("Texts classified in : %fs" % (time() - t0))


    #Scores
    predicted_classes=[]
validation_classes=[]
TP=0
FP=0
FN=0
TN=0
with open('report_SVM_lemma_BINARY_rbf_1_1_StopWordsRemoved.txt', "r") as testFile, open('/home/cjgonzal/tepeu_ayuda/training-testing/test-classes.txt')as true_classes:
    for line in testFile:
        line=line.split('\t')
        line=line[1]
        line=line.strip('\n')
        predicted_classes.append(line)
    for line in true_classes:
        line=line.strip('\n')
        validation_classes.append(line)
for pc,vc in zip(predicted_classes,validation_classes):
    if vc=='DISEASE':
        if vc==pc:
            TP+=1
        else:
            FN+=1
    elif vc=='OTHER':
        if vc==pc:
            TN+=1
        else:
            FP+=1

precission=float(TP)/(float(TP+FP))
recall=float(TP)/(float(TP+FN))
F1=2*(float(precission*recall)/float(precission+recall))

print('Precision={}\n'.format(precission))
print('Recall={}\n'.format(recall))
print('F1 score={}\n'.format(F1))
print('         | DISEASE | OTHER')
print(' DISEASE |  {}    |  {}'.format(TP,FN))
print('  OTHER  |  {}    |  {}'.format(FP,TN))
