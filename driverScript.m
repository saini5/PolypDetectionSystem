
%Driver script
%% ===============Feature Extraction=======================%%
path_folder = 'C:\Users\aman_mclarenf1\Desktop\Capstone\polyp\CVC-ClinicDB\Original\';
N = 613; % no. of images
FNAMEFMT = '%d.tif';
I = 0; 
X = []; % feature vectors of images (training set)
y = []; %cancer present or not, if present then 1, else 2.
for i=1:N
    I = imread(strcat(path_folder,sprintf(FNAMEFMT, i)));
    %imshow(I);
    %Feature Extraction Step for one image(put it in a loop for preparing
    %multiple)
    [cA,cH,cV,cD] = waveletTransform(I);
    %imshow(cV);
    %Mining features on the various domains - horizontal, vertical, diagonal,
    %overall
    statsCV = mineTexturalFeatures(cV,1);
    statsCH = mineTexturalFeatures(cH,1);
    statsCA = mineTexturalFeatures(cA,1);
    statsCD = mineTexturalFeatures(cD,1);

    %feature vectors corresponding to each wavelet transformed component
    featureVectorCV = [statsCV.Energy, statsCV.Correlation, statsCV.Homogeneity];
    featureVectorCH = [statsCH.Energy, statsCH.Correlation, statsCH.Homogeneity];
    featureVectorCA = [statsCA.Energy, statsCA.Correlation, statsCA.Homogeneity];
    featureVectorCD = [statsCD.Energy, statsCD.Correlation, statsCD.Homogeneity];

    %feature vector corresponding to the image
    featureVectorI = [featureVectorCV, featureVectorCH,featureVectorCA,featureVectorCD];
    X = [X; featureVectorI];
    
    %TODO: delete this image if you want
    label = 1;
    if i==613
        label = 2;
    end
    y = [y;label];
    
    
end

save('trainingData.mat','X','y');
pause;

%% =========================Machine learning Training===================
input_layer_size = size(X,2);
hidden_layer_size = 25;
num_labels = 2;

%Loading and visualizing data
load('trainingData.mat');  % introduced this here so anybody can separate the code for Feature Extraction and ML.
m = size(X,1);

% Initializing parameters ( weights of the neural net - it is essential to
% assign random weights as a start)
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Backpropogation
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

% Training Neural Network
%  To train the neural network, we will use "fmincg", which
%  is a function which works similarly to "fminunc". 
%  These advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

% More training can help - for that, we can increase this maxIter
options = optimset('MaxIter', 50);

%  lambda is the regularization parameter
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

             
save('trainedWeights.mat','Theta1','Theta2');
fprintf('Program paused. Press enter to continue.\n');
pause;


% Prediction
%  "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  us compute the training set accuracy.
fprintf('Loading Training Data from trainingData.mat file\n');
load('trainingData.mat');

fprintf('Loading weights trained during the training phase\n');
load('trainedWeights.mat','Theta1','Theta2');

predi = 5;
I = imread(strcat(path_folder,sprintf(FNAMEFMT, predi)));
imshow(I); pause;
pred = predict(Theta1, Theta2, X(predi,:));
fprintf('\n Prediction : %f\n',pred);

predf = 613;
I = imread(strcat(path_folder,sprintf(FNAMEFMT, predf)));
imshow(I); pause;
pred = predict(Theta1, Theta2, X(predf,:));
fprintf('\n Prediction : %f\n',pred);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


