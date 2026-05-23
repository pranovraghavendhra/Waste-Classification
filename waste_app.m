function waste_app
    modelFile = 'wasteNet_resnet18.mat';
    data = load(modelFile);
    trainedNet = data.trainedNet;

    f = figure('Name','Smart Waste Classifier','Position',[300 200 700 450],'Color',[0.95 0.95 0.95]);

    ax = axes(f,'Units','pixels','Position',[50 100 300 300]);
    title(ax,'No Image Uploaded','Color','k');

    uicontrol('Style','pushbutton','String','Upload Image','Position',[400 300 250 50],...
        'FontSize',12,'BackgroundColor',[0.2 0.6 0.8],'ForegroundColor','w','Callback',@uploadImage);

    uicontrol('Style','pushbutton','String','Classify Image','Position',[400 230 250 50],...
        'FontSize',12,'BackgroundColor',[0.3 0.7 0.3],'ForegroundColor','w','Callback',@classifyImage);

    resultText = uicontrol('Style','text','Position',[400 160 250 50],...
        'String','Prediction: ','FontSize',12,'BackgroundColor',[0.95 0.95 0.95],'ForegroundColor','k');

    barAx = axes(f,'Units','pixels','Position',[400 30 250 110]);
    title(barAx,'Confidence Scores','Color','k');

    imgData = struct('img',[],'imgPath','');

    function uploadImage(~,~)
        [file,path] = uigetfile({'*.jpg;*.png;*.jpeg','Images (*.jpg, *.png, *.jpeg)'});
        if isequal(file,0)
            return
        end
        img = imread(fullfile(path,file));
        imgData.img = img;
        imgData.imgPath = fullfile(path,file);
        imshow(img,'Parent',ax);
        title(ax,'Uploaded Image','Color','k');
        set(resultText,'String','Prediction: ');
        cla(barAx);
        set(barAx,'Color',[0.95 0.95 0.95],'XColor','k','YColor','k');
    end

    function classifyImage(~,~)
        if isempty(imgData.img)
            msgbox('Please upload an image first','Error','error');
            return
        end
        imgResized = imresize(imgData.img,trainedNet.Layers(1).InputSize(1:2));
        [label, scores] = classify(trainedNet,imgResized);
        scores = double(scores);

        [sortedScores, idx] = sort(scores,'descend');
        classes = trainedNet.Layers(end).Classes;
        top3 = classes(idx(1:3));
        top3Scores = sortedScores(1:3);

        set(resultText,'String',sprintf('Top Prediction: %s (%.2f%%)', string(top3(1)), top3Scores(1)*100));

        cla(barAx);
        bar(barAx, top3Scores*100,'FaceColor',[0.2 0.6 0.8]);
        set(barAx,'XTickLabel',cellstr(top3),'XColor','k','YColor','k','YLim',[0 100],'YGrid','on');
        ylabel(barAx,'Confidence (%)','Color','k');
        title(barAx,'Top-3 Predictions','Color','k');
    end
end
