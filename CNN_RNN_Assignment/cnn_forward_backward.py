# import libraries
import glob
import cv2
import numpy as np
import scipy.stats

# load subjective scores from dmos.mat
import scipy.io
mat = scipy.io.loadmat('dmos.mat')

# read reference images
# 28 images in this folder
img1=[]
cv_img=[]
for img in glob.glob("Live Database/databaserelease2/refimgs/*.bmp"):
		n=cv2.imread(img,0)
		img1.append(img[39:len(img)])
	    	cv_img.append(n)
# print(img1)

# read distorted images from jp2k
# 227 images in this folder
imgName1=[]
disImg1=[]
for img in glob.glob("Live Database/databaserelease2/jp2k/*.bmp"):
		if(img[36]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName1.append(img[36:len(img)])
			disImg1.append(disImgRead)
    


# read from jpeg
# 233 images in this folder
imgName2=[]
disImg2=[]
for img in glob.glob("Live Database/databaserelease2/jpeg/*.bmp"):
		if(img[36]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName2.append(img[36:len(img)])
			disImg2.append(disImgRead)



# read from wn
# 174 images in this folder
imgName3=[]
disImg3=[]
for img in glob.glob("Live Database/databaserelease2/wn/*.bmp"):
		if(img[34]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName3.append(img[34:len(img)])
			disImg3.append(disImgRead)
    


# read from gblur
# 174 images in this folder
imgName4=[]
disImg4=[]
for img in glob.glob("Live Database/databaserelease2/gblur/*.bmp"):
		if(img[37]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName4.append(img[37:len(img)])
			disImg4.append(disImgRead)


# read from fastfading
# 173 images in this folder 
# Be carefull one image is missing from folder
imgName5=[]
disImg5=[]
for img in glob.glob("Live Database/databaserelease2/fastfading/*.bmp"):
		if(img[42]=='i'):
			disImgRead=cv2.imread(img,0)
			imgName5.append(img[42:len(img)])
			disImg5.append(disImgRead)
            
# for i in range(0,len(imgName5)):
#     print(i,imgName5[i])



ysub=mat['dmos'][0]
## print subjective scores from dmos.mat 
# print(ysub[0:227])
# feature vector
X=np.zeros((982,1536,768))
# since ysub given are not according to order of ysub
# yMod is modified y's
yMod=np.zeros((982,1))
# print(X.shape)


# for entering values in X i.e cumulative feature matrix
# only for image of folder jp2k
index_X=0
f=open("Live Database/databaserelease2/jp2k/info.txt","r")
f1=f.readlines()
for x in f1:
    orgImg=x.split(" ")[0]
    dImg=x.split(" ")[1]
    if(dImg[5]=='b'):
        p=dImg[3]
        p=int(p,10)
        yMod[index_X,:]=ysub[p-1]
    elif(dImg[6]=='b'):
        p=dImg[3:5]
        p=int(p,10)
        yMod[index_X,:]=ysub[p-1]
    elif(dImg[7]=='b'):
        p=dImg[3:6]
        p=int(p,10)
        yMod[index_X,:]=ysub[p-1]

#   original image matrix
    k=img1.index(orgImg)
    A=cv_img[k]
    (m,n)=A.shape

#   perturbed image matrix
    l=imgName1.index(dImg)
    Ap=disImg1[l]
    # print(Ap.shape)
    (x,y)=Ap.shape

#   cummulating image and its perturbed version
    X[index_X,0:x,0:y]=Ap;
    X[index_X,768:m+768,0:n]=A;
#     print(index_X,X[0:226])
    index_X=index_X+1


# for jpeg folder
f=open("Live Database/databaserelease2/jpeg/info.txt","r")
f1=f.readlines()
for x in f1:
        orgImg=x.split(" ")[0]
        dImg=x.split(" ")[1]
        offset1=226
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+226]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+226]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+226]
            
#   for original image
        k=img1.index(orgImg)
        A=cv_img[k]
        (m,n)=A.shape
        
#   for distorted image
        l=imgName2.index(dImg)
        Ap=disImg2[l]
        (x,y)=Ap.shape
      
        X[index_X,0:x,0:y]=Ap;
        X[index_X,768:m+768,0:n]=A;
        index_X=index_X+1

# print(X[450,:,:],index_X)
# for wind noise data
f=open("Live Database/databaserelease2/wn/info.txt","r")
f1=f.readlines()
for x in f1:
        orgImg=x.split(" ")[0]
        dImg=x.split(" ")[1]
        # print(dImg,orgImg)
        offset2=459
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+459]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+459]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+459]
        
#         for original image
        k=img1.index(orgImg)
        A=cv_img[k]
        (m,n)=A.shape
        
#         for distorted image
        l=imgName3.index(dImg)
        Ap=disImg3[l]
        (x,y)=Ap.shape
        
        X[index_X,0:x,0:y]=Ap;
        X[index_X,768:m+768,0:n]=A;
        index_X=index_X+1

# for gblur data
# print(index_X)
f=open("Live Database/databaserelease2/gblur/info.txt","r")
f1=f.readlines()
# print(f1)
# print(f1)
for x in f1:
    for i in range(0,174):
        temp=x.split("\r\r")[i]
#         print(temp)
        orgImg=temp.split(" ")[0]
        dImg=temp.split(" ")[1]
#         print(i,orgImg,dImg)
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+633]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+633]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
            yMod[index_X,:]=ysub[p+633]
            
        k=img1.index(orgImg)
        A=cv_img[k]
        (m,n)=A.shape
        
        l=imgName4.index(dImg)
        Ap=disImg4[l]
        (x1,y1)=Ap.shape
        
        X[index_X,0:x1,0:y1]=Ap;
        X[index_X,768:m+768,0:n]=A;
        index_X=index_X+1

# for fastfading data
f=open("Live Database/databaserelease2/fastfading/info.txt","r")
f1=f.readlines()
for x in f1:
        orgImg=x.split(" ")[0]
        dImg=x.split(" ")[1]
        offset1=807
        if(dImg[5]=='b'):
            p=dImg[3]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+807]
        elif(dImg[6]=='b'):
            p=dImg[3:5]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+807]
        elif(dImg[7]=='b'):
            p=dImg[3:6]
            p=int(p,10)
    #         print(p)
            yMod[index_X,:]=ysub[p+807]
        if(dImg!="img1.bmp"):
            k=img1.index(orgImg)
            
#             for original image
            A=cv_img[k]
            (x1,y1)=A.shape
#     for distorted image
            l=imgName5.index(dImg)
            Ap=disImg5[l]
            (m,n)=Ap.shape
            index_X=index_X+1

# print the final image matrix
# containing pixel as a features
# X(index_X,0:768,:) contain original image 
# while X(index_X,768:1536,:) contain perturbed version
# print(X.shape)
# print(X[1,769,:])
# yMod is matrix containing similarity values
# print(yMod)

# perform convolution to predict the result
#single step convolution
#returns a singular value after convolution
def conv_single_step(A_slice, W):
    s = np.multiply(A_slice,W)
    Z = np.sum(s) 
    return Z

def forward_feed(A,W): 
    # every filter is of size f*f*n_c
    # calculating shapes of image matrix and weight matrix
    (no_of_example,horiz_size,ver_size)=A.shape
    (f,f)=W.shape
    stride=5
    
    #calculating dimension of output matrix after convolution
    out_hori_size=int((horiz_size - f)/stride) + 1
    out_ver_size=int((ver_size - f)/stride) + 1
    output_after_conv=np.zeros((no_of_example,out_hori_size,out_ver_size))
    
    no_of_example=10
    
    for index in range(no_of_example):
        original_image=X[index,0:768,:]
        perturbed_image=X[index,768:1536,:]
        #conv operation for original image
        for horiz in range(int((768-f)/stride)+1):
            for vert in range(int((768-f)/stride)+1):
                horiz_start=horiz*stride
                horiz_end=horiz_start+f
                print(horiz_end-horiz_start)
                vert_start=vert*stride
                vert_end=vert_start+f
                print(vert_end-vert_start)
                
                slice_org = original_image[horiz_start:horiz_end, vert_start:vert_end]
                print(slice_org.shape,index)
                output_after_conv[index, horiz, vert] = conv_single_step(slice_org, W)
                
                slice_pertubed=perturbed_image[horiz_start:horiz_end,vert_start:vert_end]
                print(slice_pertubed.shape,index)
                output_after_conv[index,horiz+int((768-f)/stride)+1,vert]=conv_single_step(slice_pertubed,W)
    
    return output_after_conv       

z = forward_feed(X,W)
print(z.shape)

#Add max pooling layer
def max_pool(A): 
    # every filter is of size f*f*n_c
    # calculating shapes of image matrix 
    (no_of_example,horiz_size,ver_size)=A.shape
    
    #hyperparameters are padding=0 and stride=5
    stride=2
    filter_size=10
    
    #calculating dimension of output matrix after max pooling
    out_hori_size=int((horiz_size - filter_size)/stride) + 1
    out_ver_size=int((ver_size - filter_size)/stride) + 1
    output_max_pool=np.zeros((no_of_example,out_hori_size,out_ver_size))
    
    no_of_example=10
    
    for index in range(no_of_example):
        #conv operation for original image
        for horiz in range(out_hori_size):
            for vert in range(out_ver_size):
                horiz_start=horiz*stride
                horiz_end=horiz_start+filter_size
                
                vert_start=vert*stride
                vert_end=vert_start+filter_size
                
                slice_image = A[horiz_start:horiz_end, vert_start:vert_end]
                              
                output_max_pool[index, horiz, vert] = np.max(slice_image)
    
    return output_max_pool 
    
# obtain matrix after max pooling 
# reshape the matrix which act as feature vector
max_pooling=max_pool(z)
(no_of_images,size1,size2)=max_pooling.shape

reshape_fc_vector=np.zeros((no_of_images,size1*size2))
for index in range(no_of_images):
    reshape_fc_vector[index,:]=np.reshape(max_pooling[index,:,:],max_pooling[index].size)

print(reshape_fc_vector.shape)

#Create a random weight matrix for fully connected layer
(no_of_ex,fully_connected_layer_size)=reshape_fc_vector.shape
weight_fully_connected=np.random.randn(fully_connected_layer_size,1)


from scipy.stats import logistic
#predict the output and calculate error using mean square error
Y=np.dot(reshape_fc_vector,weight_fully_connected)
# Applying sigmoid activation function to predict Y
error=np.zeros((no_of_ex,1))
Y_sigmoid=np.zeros((no_of_ex,1))
for i in range(Y.size):
    Y_sigmoid[i,0]=logistic.cdf(Y[i,0])
    error[i,0]=np.square(Y_sigmoid[i,0]-yMod[i,0])*0.5

# sum all the error and iterate till error is minimized
Error=np.sum(error)
print(Error)


# perform backpropagration and update fully connected layer
# dE/dW=(Y_sigmoid-yMod)*(Y_sigmoid(1-Y_sigmoid))*reshape_fc_vector
# grgradient_fully_connected=np.multiply(np.multiply((Y_sigmoid-yMod),np.multiply(Y_sigmoid,(1-Y_sigmoid))),reshape_fc_vector)
# print(gradient_fully_connected.shape)

def conv_backward(dZ, cache):
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride

                    vert_end = vert_start + f
                    horiz_start = w * stride

                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    
    return dA_prev, dW, db


def pool_backward(dA, cache, mode = "max"):
    
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       # loop over the training examples
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                    
    
    
    return dA_prev