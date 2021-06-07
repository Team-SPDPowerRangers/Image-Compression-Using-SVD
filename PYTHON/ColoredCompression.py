import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix

def newMatrixSVDandError(num):
    global dividingFactor
    global U_red, S_red, VT_red
    global U_green, S_green, VT_green
    global U_blue, S_blue, VT_blue
    global displayErrorRED,singularValsRED
    global displayErrorGREEN,singularValsGREEN
    global displayErrorBLUE,singularValsBLUE
    global displayErrorCOLOR,singularValsCOLOR
    N=num
        # on red component

    U_RED = U_red[:, :N]
    S_RED = np.diag(S_red[:N])
    VT_RED = VT_red[:N, :]
    D_RED = U_RED @ S_RED @ VT_RED
    if N<=40:
        print("The S matrix here is a sparse matrix of ", N, " singular values")
        print("Normal representation : ")
        print(S_RED)
        print("This is sparse representation : ")
        print(dia_matrix(S_RED))
    zeros_red = np.zeros([np.shape(D_RED)[0], np.shape(D_RED)[1]])
    RimgD = cv2.merge((zeros_red, zeros_red, D_RED))
    Rimg = np.uint8(RimgD)
    Rimg_Row_scale = int(np.shape(Rimg)[0] / dividingFactor)
    Rimg_COL_scale = int(np.shape(Rimg)[1] / dividingFactor)
    RimgResize = cv2.resize(Rimg, (Rimg_Row_scale, Rimg_COL_scale))
    cv2.imshow(('Red with ' + str(N) + ' singular values'), RimgResize)
    cv2.waitKey(0)
    cv2.imwrite(('Red Component - ' + str(N) + '.jpg'), Rimg)
    errorRED = sum(sum((Red - D_RED) ** 2))
    displayErrorRED.append(errorRED)
    singularValsRED.append(N)

        # on green component
    U_GREEN = U_green[:, :N]
    S_GREEN = np.diag(S_green[:N])
    VT_GREEN = VT_green[:N, :]
    D_GREEN = U_GREEN @ S_GREEN @ VT_GREEN
    if N<=40:
        print("The S matrix here is a sparse matrix of ", N, " singular values")
        print("Normal representation : ")
        print(S_GREEN)
        print("This is sparse representation : ")
        print(dia_matrix(S_GREEN))
    zeros_green = np.zeros([np.shape(D_GREEN)[0], np.shape(D_GREEN)[1]])
    GimgD = cv2.merge((zeros_green, D_GREEN, zeros_green))
    Gimg = np.uint8(GimgD)
    Gimg_ROW_scale = int(np.shape(Gimg)[0] / dividingFactor)
    Gimg_COL_scale = int(np.shape(Gimg)[1] / dividingFactor)
    GimgResize = cv2.resize(Gimg, (Gimg_ROW_scale, Gimg_COL_scale))
    cv2.imshow(('Green with ' + str(N) + ' singular values'), GimgResize)
    cv2.waitKey(0)
    cv2.imwrite(('Green with - ' + str(N) + '.jpg'), Gimg)
    errorGREEN = sum(sum((Green - D_GREEN) ** 2))
    displayErrorGREEN.append(errorGREEN)
    singularValsGREEN.append(N)
    # on blue component

    U_BLUE = U_blue[:, :N]
    S_BLUE = np.diag(S_blue[:N])
    VT_BLUE = VT_blue[:N, :]
    D_BLUE = U_BLUE @ S_BLUE @ VT_BLUE
    if N<=40:
        print("The S matrix here is a sparse matrix of ", N, " singular values")
        print("Normal representation : ")
        print(S_BLUE)
        print("This is sparse representation : ")
        print(dia_matrix(S_BLUE))
    zeros_blue = np.zeros([np.shape(D_BLUE)[0], np.shape(D_BLUE)[1]])
    BimgD = cv2.merge((D_BLUE, zeros_blue, zeros_blue))
    Bimg = np.uint8(BimgD)
    Bimg_ROW_scale = int(np.shape(Bimg)[0] / dividingFactor)
    Bimg_COL_scale = int(np.shape(Bimg)[1] / dividingFactor)
    BimgResize = cv2.resize(Bimg, (Bimg_ROW_scale, Bimg_COL_scale))
    cv2.imshow(('Blue with ' + str(N) + ' singular values'), BimgResize)
    cv2.waitKey(0)
    cv2.imwrite(('Blue component - ' + str(N) + '.jpg'), Bimg)
    errorBLUE = sum(sum((Blue - D_BLUE) ** 2))
    displayErrorBLUE.append(errorBLUE)
    singularValsBLUE.append(N)

        # getting back colored image
    CimgD = cv2.merge((D_BLUE, D_GREEN, D_RED))
    Cimg = np.uint8(CimgD)
    Cimg_ROW_scale = int(np.shape(Cimg)[0] / dividingFactor)
    Cimg_COL_scale = int(np.shape(Cimg)[1] / dividingFactor)
    CimgResize = cv2.resize(Cimg, (Cimg_ROW_scale, Cimg_COL_scale))
    cv2.imshow(('Colored with ' + str(N) + ' singular values'), CimgResize)
    cv2.waitKey(0)
    cv2.imwrite(('RGB component - ' + str(N) + '.jpg'), Cimg)



filename=input("Enter file name : ")
filename='../'+filename

inImage=cv2.imread(filename)
[imRow,imCol,imH]=np.shape(inImage)
print("Image size is : ",imRow,'x',imCol)
dividingFactor=int(input("Enter dividing factor : "))
imShowingRowScale=int(imRow/dividingFactor)
imShowingColScale=int(imCol/dividingFactor)
inImageResize=cv2.resize(inImage, (imShowingRowScale,imShowingColScale))
cv2.imshow('Original Image',inImageResize)
cv2.imwrite('Original COLORED Image.jpg',inImage)
cv2.waitKey(0)

displayErrorRED=[]
singularValsRED=[]
displayErrorGREEN=[]
singularValsGREEN=[]
displayErrorBLUE=[]
singularValsBLUE=[]

zeros=np.zeros(inImage.shape[:2],dtype="uint8")

B,G,R=cv2.split(inImage)

Rimg=cv2.merge([zeros,zeros,R])
inImageRedResize=cv2.resize(Rimg, (imShowingRowScale,imShowingColScale))
cv2.imshow('RED',inImageRedResize)
cv2.imwrite('Original RED Image.jpg',Rimg)
cv2.waitKey(0)

Gimg=cv2.merge([zeros,G,zeros])
inImageGreenResize=cv2.resize(Gimg, (imShowingRowScale,imShowingColScale))
cv2.imshow('GREEN',inImageGreenResize)
cv2.imwrite('Original GREEN Image.jpg',Gimg)
cv2.waitKey(0)

Bimg=cv2.merge([B,zeros,zeros])
inImageBlueResize=cv2.resize(Bimg, (imShowingRowScale,imShowingColScale))
cv2.imshow('BLUE',inImageBlueResize)
cv2.imwrite('Original BLUE Image.jpg',Bimg)
cv2.waitKey(0)

cv2.destroyAllWindows()


Red=np.double(R)
Green=np.double(G)
Blue=np.double(B)
U_red, S_red, VT_red = np.linalg.svd(Red)
U_green, S_green, VT_green = np.linalg.svd(Green)
U_blue, S_blue, VT_blue = np.linalg.svd(Blue)

newMatrixSVDandError(1)
cv2.destroyAllWindows()


startRange=int(input('Enter minimum singular values : '))
endRange=int(input('Enter maximum singular values : '))
stepSize=int(input('Enter step size : '))

for N in range(startRange, endRange+2, stepSize):
    newMatrixSVDandError(N)
cv2.destroyAllWindows()

def plotGraphs():
    global displayErrorRED, singularValsRED
    global displayErrorGREEN, singularValsGREEN
    global displayErrorBLUE, singularValsBLUE
    global displayErrorCOLOR, singularValsCOLOR
        # Plot RED error
    plt.figure('RED')
    plt.title('Error in compression')
    plt.plot(singularValsRED, displayErrorRED, 'r')
    plt.grid('on')
    plt.xlabel('Number of Singular Values used')
    plt.ylabel('Error between compress and original image')
        # Plot GREEN error
    plt.figure('GREEN')
    plt.title('Error in compression')
    plt.plot(singularValsGREEN, displayErrorGREEN, 'g')
    plt.grid('on')
    plt.xlabel('Number of Singular Values used')
    plt.ylabel('Error between compress and original image')
        # Plot BLUE error
    plt.figure('BLUE')
    plt.title('Error in compression')
    plt.plot(singularValsBLUE, displayErrorBLUE, 'b')
    plt.grid('on')
    plt.xlabel('Number of Singular Values used')
    plt.ylabel('Error between compress and original image')

plotGraphs()

# subplot representation
plt.figure('ALL IN ONE')
plt.title('Error in compression')
plt.subplot(1,3,1)
plt.plot(singularValsRED,displayErrorRED,'r')
plt.ylabel('Error between compress and original image')
plt.title('Error in RED')
plt.grid('on')
plt.subplot(1,3,2)
plt.plot(singularValsGREEN,displayErrorGREEN,'g')
plt.xlabel('Number of Singular Values used')
plt.title('Error in GREEN')
plt.grid('on')
plt.subplot(1,3,3)
plt.plot(singularValsBLUE,displayErrorBLUE,'b')
plt.title('Error in BLUE')
plt.grid('on')
plt.suptitle('Errors of each color components')
plt.show()

print("Done !!")


'''
 ## PSEUDO CODE OF COLORED IMAGE COMPRESSION
    -> Original Image Input
    -> Split into RGB components
    -> Perform SVD
      -> SVD on RED
        -> New RED matrix  
      -> SVD on GREEN
        -> New GREEN matrix    
      -> SVD on BLUE
        -> New BLUE matrix
    -> New COLOR Image
      -> Combine New RED, GREEN & BLUE
    -> Keep necessary value
    -> Unnecessary values are made zero(SPARSE)
    -> SVD on given range
    -> Display and store images
    -> Show error  
'''