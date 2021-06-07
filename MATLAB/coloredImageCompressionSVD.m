clc; clear all; close all;

filename=input('Enter file name (with"")      : ');
filename=strcat('../',filename);
[X,map]=imread(filename);
figure('name','Original Colored Image')
imshow(X);
title('Original Image')
imwrite(X,'Original Colored Image.jpg');

R =X(:,:,1);
G =X(:,:,2);
B =X(:,:,3);    
Rimg=cat(3,R,zeros(size(R)),zeros(size(R)));     
Gimg=cat(3,zeros(size(G)),G,zeros(size(G)));    
Bimg=cat(3,zeros(size(B)),zeros(size(B)),B);

figure('name','RGB color components separated')    
    % Red component
subplot(1,3,1)
imshow(Rimg)
title('R component')
imwrite(Rimg,'Original Red.jpg')
    % Green component
subplot(1,3,2)
imshow(Gimg)
title('G component')
imwrite(Gimg,'Original Green.jpg')
    % Blue component
subplot(1,3,3)
imshow(Bimg)
title('B component')
imwrite(Bimg,'Original Blue.jpg')
suptitle('Separated R G B Components')


% Making double datatype for SVD
Red=double(R);
Green=double(G);
Blue=double(B);

displayErrorRED=[];
singularValsRED=[];
%%%%%%%%%%%% RED IMAGE
[U_red,S_red,V_red]=svd(Red);
rank(S_red);
N=1;
C_red=S_red;
C_red(N+1:end,:)=0;
C_red(:,N+1:end)=0;
D_red=U_red*C_red*V_red';    
figure('name',sprintf('Images with %d singular values',N));
buffer = sprintf('Red image output using %d singular values', N);
Rimg=cat(3,D_red,zeros(size(D_red)),zeros(size(D_red)));
new_Rimg=uint8(Rimg);
subplot(2,2,1)
imshow(new_Rimg)
imwrite(new_Rimg,strcat(buffer,'.jpg'));
title(sprintf('R Component- %d ',N))
error_red=sum(sum((Red-D_red).^2));
displayErrorRED=[displayErrorRED;error_red];
singularValsRED=[singularValsRED;N];


displayErrorGREEN=[];
singularValsGREEN=[];
%%%%%%% GREEN IMAGE
[U_green,S_green,V_green]=svd(Green);
C_green=S_green;
C_green(N+1:end,:)=0;
C_green(:,N+1:end)=0;
D_green=U_green*C_green*V_green';
buffer = sprintf('Green image output using %d singular values', N);
Gimg=cat(3,zeros(size(D_green)),D_green,zeros(size(D_green)));
new_Gimg=uint8(Gimg);
subplot(2,2,2)
imshow(new_Gimg)
imwrite(new_Gimg,strcat(buffer,'.jpg'));
title(sprintf('G Component- %d ',N))
error_green=sum(sum((Green-D_green).^2));
displayErrorGREEN=[displayErrorGREEN;error_green];
singularValsGREEN=[singularValsGREEN;N];

displayErrorBLUE=[];
singularValsBLUE=[];
%%%%%%% BLUE IMAGE
[U_blue,S_blue,V_blue]=svd(Blue);
C_green=S_blue;
C_green(N+1:end,:)=0;
C_green(:,N+1:end)=0;
D_blue=U_blue*C_green*V_blue';
    % get back the blue image after SVD
%figure('name','Blue image after SVD')
buffer = sprintf('Blue image output using %d singular values', N);
Bimg=cat(3,zeros(size(D_blue)),zeros(size(D_blue)),D_blue);
new_Bimg=uint8(Bimg);
subplot(2,2,3)
imshow(new_Bimg)
imwrite(new_Bimg,strcat(buffer,'.jpg'));
title(sprintf('B Component- %d ',N))
error_blue=sum(sum((Blue-D_blue).^2));
displayErrorBLUE=[displayErrorBLUE;error_blue];
singularValsBLUE=[singularValsBLUE;N];



%%%%% COMBINING THESE THREE BACK
buffer = sprintf('Colored image output using %d singular values', N);
Cimg=cat(3,D_red,D_green,D_blue);
new_Cimg=uint8(Cimg);
subplot(2,2,4)
imshow(new_Cimg)
imwrite(new_Cimg,strcat(buffer,'.jpg'));
title(sprintf('RGB Component- %d ',N))
suptitle(sprintf('R G B Images with %d singular values',N))


min=input('Enter minimum singular values : ');
max=input('Enter maximum singular values : ');
step=input('Enter step size : ');

for N=min:step:max

    figure('name',sprintf('Images with %d singular values',N));
    
    % Recompute modes for the red image - already solved by SVD above
    C_red=S_red;
    C_red(N+1:end,:)=0;
    C_red(:,N+1:end)=0;
    D_red=U_red*C_red*V_red';
        % Rebuild the data back into a displayable image and show it
    %figure;
    %buffer = sprintf('Red image output using %d singular values', N);
    buffer=sprintf('R Component - %d',N);
    Rimg = cat(3, D_red, zeros(size(D_red)), zeros(size(D_red)));
    new_Rimg=uint8(Rimg);
    subplot(2,2,1)
    imshow(new_Rimg);
    imwrite(new_Rimg, strcat(buffer,'.jpg'));
    title(buffer);
    error_red=sum(sum((Red-D_red).^2));
    displayErrorRED=[displayErrorRED;error_red];
    singularValsRED=[singularValsRED;N];

    
    % Recompute modes for the green image - already solved by SVD above
    C_green = S_green;
    C_green(N+1:end,:)=0;
    C_green(:,N+1:end)=0;
    D_green=U_green*C_green*V_green';
    % Rebuild the data back into a displayable image and show it
    %figure;
    %buffer = sprintf('Green image output using %d singular values', N);
    buffer=sprintf('G Component - %d',N);
    Gimg = cat(3, zeros(size(D_green)), D_green, zeros(size(D_green)));
    new_Gimg=uint8(Gimg);
    subplot(2,2,2);
    imshow(new_Gimg);    
    imwrite(new_Gimg ,strcat(buffer,'.jpg'));
    title(buffer);
    error_green=sum(sum((Green-D_green).^2));
    displayErrorGREEN=[displayErrorGREEN;error_green];
    singularValsGREEN=[singularValsGREEN;N];
    
    % Recompute modes for the blue image - already solved by SVD above
    C_blue = S_blue;
    C_blue(N+1:end,:)=0;
    C_blue(:,N+1:end)=0;
    D_blue=U_blue*C_blue*V_blue';
    % Rebuild the data back into a displayable image and show it
    %figure;
    %buffer = sprintf('Blue image output using %d singular values', N);
    buffer=sprintf('B Component - %d',N);
    Bimg = cat(3, zeros(size(D_blue)), zeros(size(D_blue)), D_blue);
    new_Bimg=uint8(Bimg);
    subplot(2,2,3);
    imshow(new_Bimg);    
    imwrite(new_Bimg, strcat(buffer,'.jpg'));
    title(buffer);
    error_blue=sum(sum((Blue-D_blue).^2));
    displayErrorBLUE=[displayErrorBLUE;error_blue];
    singularValsBLUE=[singularValsBLUE;N];
    
    % Take the data from the Red, Green, and Blue image
    % Rebuild a colored image with the corresponding data and show it
    %figure;
    %buffer = sprintf('Colored image output using %d singular values', N);
    buffer=sprintf('RGB Component - %d',N);
    Cimg = cat(3, D_red, D_green, D_blue);
    new_Cimg=uint8(Cimg);
    subplot(2,2,4);
    imshow(new_Cimg);
    imwrite(new_Cimg, strcat(buffer,'.jpg'));
    title(buffer);
    suptitle(sprintf('RGB Images with %d singular values',N))
    
end

figure('name','Error Vs Singular Values');
hold on;
plot(singularValsRED, displayErrorRED,'r');
plot(singularValsGREEN, displayErrorGREEN,'g');
plot(singularValsBLUE, displayErrorBLUE,'b');
grid on
xlabel('Number of Singular Values used');
ylabel('Error between compress and original image');
title('RGB Error in compression');
legend('RED','GREEN','BLUE')


fprintf('Done !!!\n')