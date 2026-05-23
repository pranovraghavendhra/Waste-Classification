function label = classify_image(imagePath, modelFile)

if nargin<2, modelFile = fullfile(pwd,'wasteNet_resnet18.mat'); end
S = load(modelFile, 'trainedNet');
net = S.trainedNet;
img = imread(imagePath);


if size(img,3)==1
    img = cat(3,img,img,img);
end

img = imresize(img, net.Layers(1).InputSize(1:2));
label = classify(net, img);
end
