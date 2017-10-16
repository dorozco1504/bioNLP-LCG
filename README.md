# bioNLP-LCG
Este grupo forma parte de la Licenciatura en Ciencias Genómicas de la UNAM, donde se hizo necesario el desarrollo de un proyecto bioinformático, que a grandes rasgos, abarca la construcción de una base de datos, la cual hace uso de la ciencia de la minería de textos y el procesamiento del lenguaje natural, para así conseguir la extracción de relación entre polimorfismo/ enfermedad, esto a partir de 20649 resúmenes de artículos biomédicos extraídos de disGenet. para tal objetivo se hizo uso de 3 clasificadores distintos, VPN, Perceptrón y Árboles aleatorios, los cuales fueron entrenados para su posterior evaluación por medio de validación cruzada. 
Por último los mejores resultados fueron seleccionados para su prueba en el problema inicial. 
El presente repositorio contiene:
1. Training-cross-validation-improving.py.
  Código con el que se implementó el entrenamiento de los clasificadores y la validación cruzada.
2. Testing.py que se implementó 
  Código que se implementó para procesar los datos de prueba con los mejores modelos obtenidos en la fase de entrenamiento. 
