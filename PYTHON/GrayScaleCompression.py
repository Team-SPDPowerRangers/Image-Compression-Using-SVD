import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix

def newMatrixAfterSVDandErrorAddition(num):
    global displayError,singularVals,inImageD
    N=num
    gray_U = U[:, :N]       # mxm
    gray_S = np.diag(S[:N]) # mxn
    gray_VT = VT[:N, :]     # nxn
    DN = gray_U @ gray_S @ gray_VT
    if N<=40:
        print("The S matrix here is a sparse matrix of ", N, " singular values")
        print("Normal representation : ")
        print(gray_S)
        print("This is sparse representation : ")
        print(dia_matrix(gray_S))
    D = np.uint8(DN)
    DImageResize = cv2.resize(D, (imShowingRowScale, imShowingColScale))
    cv2.imshow(('Image with ' + str(N) + ' singular values'), DImageResize)
    cv2.waitKey(0)
    cv2.imwrite(('Gray Image - '+str(N)+'.jpg'), D)
    error = sum(sum((inImageD - DN) ** 2)) # sum of squared errors
    displayError.append(error)
    singularVals.append(N)

def visualizeError():
    plt.figure('GRAPHICAL COMPARISON')
    plt.title('ERROR IN COMPRESSION')
    plt.plot(singularVals, displayError, 'k')
    plt.grid('on')
    plt.xlabel('Number of Singular Values used')
    plt.ylabel('Error between compress and original image')
    plt.show()

filename=input("Enter file name : ")
filename='../'+filename
inImage=cv2.imread(filename)

[imRow, imCol, imH] = np.shape(inImage)
print("Image size is : ",imRow,'x',imCol)
dividingFactor=int(input("Enter dividing factor : "))
imShowingRowScale=int(imRow/dividingFactor)
imShowingColScale=int(imCol/dividingFactor)
inImageResize = cv2.resize(inImage, (imShowingRowScale, imShowingColScale))

cv2.imshow('Original Image', inImageResize)
cv2.waitKey(0)
cv2.imwrite('Original Image.jpg', inImage)

inImage = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY)

inImageResize = cv2.resize(inImage, (imShowingRowScale, imShowingColScale))
cv2.imshow('Gray_Scaled Image', inImageResize)
cv2.waitKey(0)
cv2.imwrite('Original Gray Image.jpg', inImage)


displayError= []
singularVals = []

inImageD = np.double(inImage)
[U, S, VT] = np.linalg.svd(inImageD, full_matrices=True)

# perform svd for N=1
newMatrixAfterSVDandErrorAddition(1)
cv2.destroyAllWindows()

startRange=int(input('Enter minimum singular values : '))
endRange=int(input('Enter maximum singular values : '))
stepSize=int(input('Enter step size : '))

for N in range(startRange, endRange+1, stepSize):
    newMatrixAfterSVDandErrorAddition(N)

cv2.destroyAllWindows()
visualizeError()
print("Done !!")


'''
  ## PSEUDO CODE OF GRAY IMAGE COMPRESSION
    -> Original image input
    -> Convert to gray scale
    -> Perform SVD
    -> Keep necessary value
    -> Unnecessary values are made zero(SPARSE)
    -> SVD on given range
    -> Display and store images
    -> Show error 
'''