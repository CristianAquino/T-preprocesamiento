# Importacion de librerias
import cv2
import numpy as np
import imutils

# lectura de la imagen
original = cv2.imread('H-1.jpg')
# ajustando el tamaño de la imagen
# original = imutils.resize(original, width=100)

# informacion de la imagen
(alto, ancho, x) = original.shape
print('alto: {} pixeles'.format(original.shape[0]))
print('ancho: {} pixeles'.format(original.shape[1]))

# reduccion del brillo de la imagen


def ajusteBrillo(image, gamma=1.0):
    invGama = 1.0/gamma
    table = np.array(
        [((i/255.0)**invGama)*255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)


# realizando ajuste del brillo
brillo = ajusteBrillo(original, 0.3)
cv2.imshow('Imagen Original - Brillo Reducido', np.hstack([original, brillo]))
cv2.waitKey(0)

# aplicacion del filtro GaussianBlur
gauss = cv2.GaussianBlur(brillo, (5, 5), 0)

# conversion de la imagen a HSV para una mejor deteccion con respecto al fondo
hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)
cv2.imshow('Filtro Gaussian - HSV', np.hstack([gauss, hsv]))
cv2.waitKey(0)

# conversion de la imagen a escala de grises
gris = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
cv2.imshow('gris', gris)
cv2.waitKey(0)

# umbralizacion con el metodo otsu
_, otsu = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
cv2.imshow('Otsu - gris', np.hstack([otsu, gris]))
cv2.waitKey(0)

# aplicacion del metodo otsu en la imagen original
result = cv2.bitwise_and(original, original, mask=otsu)
cv2.imshow('result', result)
cv2.waitKey(0)

# dibujando contorno
contorno, _ = cv2.findContours(
    otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result, contorno, -1, (0, 255, 0), 3, cv2.LINE_AA)
cv2.imshow('Otsu - gris', np.hstack([result, original]))
cv2.waitKey(0)
