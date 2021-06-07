clc; clear all; close all;

filename=input('Enter file name (with"")      : ');
filename=strcat('../',filename);
inImage=imread(filename);
figure('name','Original Colored Image')
imshow(filename)
title('Original Image')
imwrite(inImage,'Original Colored Image.jpg')


inImage=rgb2gray(inImage);
figure('name','Original image in GreyScale')
imshow(inImage)
title('Gray Scaled Image')
name=strcat('Original Gray Scaled','.jpg');
imwrite(inImage,name)


inImageD=double(inImage);
[U,S,V]=svd(inImageD);

displayError=[];  
singularVals=[];

N=1;    
C=S;    
C(N+1:end,:)=0;
C(:,N+1:end)=0;
D=U*C*V' ;
error=sum(sum((inImageD-D).^2));
displayError=[displayError;error];
singularVals=[singularVals;N];

figure('name','Image with 1 singular value')
buffer = sprintf('Gray Image output using %d singular values', N);
imshow(uint8(D))
newImage=uint8(D);    
imwrite(newImage,strcat(buffer,'.jpg'))
title(buffer);

min=input('Enter minimum singular values : ');
max=input('Enter maximum singular values : ');
step=input('Enter step size : ');

for N=min:step:max
    C=S;
    C(N+1:end,:)=0;
    C(:,N+1:end)=0;
    D=U*C*V';    
    error=sum(sum((inImageD-D).^2));
    displayError=[displayError;error];
    singularVals=[singularVals;N];    
    buffer = sprintf('GrayScale - %d', N);
    figure('name',buffer)
    new_img=uint8(D);
    imshow(new_img);
    imwrite(new_img, strcat(buffer,'.jpg'));
    title(buffer);    
end

figure;
plot(singularVals, displayError);
grid on
title('Error In Compression');
xlabel('Number of Singular Values used');
ylabel('Error between compressed and original image');
fprintf("Done !!!\n")

