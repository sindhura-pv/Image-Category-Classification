url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
outputFolder = 'C:\Users\sindhura\Desktop\nam pics\caltech101\101_ObjectCategories';
categories = {'airplanes', 'ferry', 'laptop'};
imds = imageDatastore(fullfile(outputFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
% Notice that each set now has exactly the same number of images.
airplanes = find(imds.Labels == 'airplanes', 1);
ferry = find(imds.Labels == 'ferry', 1);
laptop = find(imds.Labels == 'laptop', 1);
figure
subplot(1,3,1);
imshow(readimage(imds,airplanes))
subplot(1,3,2);
imshow(readimage(imds,ferry))
subplot(1,3,3);
imshow(readimage(imds,laptop))

net = alexnet();
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');



w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')

featureLayer = 'fc7';
%trainingFeatures = activations(net, trainingSet, featureLayer, ...
%   'MiniBatchSize', 4, 'OutputAs', 'columns');
trainingLabels = trainingSet.Labels;
 % Extract test features using the CNN
%testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

mean1 = mean(diag(confMat)); 
%Evaluating the classifier
newImage = fullfile(outputFolder, 'laptop', 'image_0071.jpg');

% Pre-process the images as required for the CNN
img = readAndPreprocessImage(newImage);

% Extract image features using the CNN
imageFeatures = activations(net, img, featureLayer);
% Make a prediction using the classifier
label = predict(classifier, imageFeatures);
disp(label);

function Iout = readAndPreprocessImage(filename)
        I = imread(filename);
        if ismatrix(I)
            I = cat(3,I,I,I);
        end
        Iout = imresize(I, [227 227]);
end