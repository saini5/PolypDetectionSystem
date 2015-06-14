function [cA,cH,cV,cD] = waveletTransform(x)
    
%     imshow(x);
%     pause;
    x = rgb2gray(x); %converting to gray
    [cA,cH,cV,cD] = dwt2(x,'haar');
%     imshow(cA); pause;
%     imshow(cH); pause;
%     imshow(cV); pause;
%     imshow(cD); pause;
end