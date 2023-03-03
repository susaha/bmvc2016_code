%------------------------------------------------------------------------
% Author: Gurkirt Singh
% The following script checks for an flow image which has very small optical
% flow fields and replaces that with near by flow image.
%------------------------------------------------------------------------
function processMotionLessFrames()
clc
clear
basedir = '/mnt/sun-gamma/ss-workspace/JHMDB21_data/';
cvofdir = [basedir,'cvof/'];
newCvofdir = [basedir,'cvof-modified/'];
actionlist = dir(cvofdir);
for actionNum = 3:length(actionlist)
    actiondir = [cvofdir,actionlist(actionNum).name,'/'];
    newactiondir = [newCvofdir,actionlist(actionNum).name,'/'];
    videolist = dir(actiondir);
    for vidNum = 3:length(videolist)
        
        imgdir = [actiondir,videolist(vidNum).name,'/'];
        newimgdir = [newactiondir,videolist(vidNum).name,'/'];
        fprintf('processing action %d  video %d\n',actionNum-2,vidNum-2);
        
        if ~exist(newimgdir,'dir')
            mkdir(newimgdir)
        end
        processOneVideo(imgdir,newimgdir)
        
    end
end

end

function processOneVideo(imgdir,newimgdir)

imagelist = dir([imgdir,'*.png']);
numFrames = length(imagelist);
vars = zeros(numFrames,1);
for imgNum = 1: length(imagelist)
    image = imread([imgdir,imagelist(imgNum).name]);
    
    % computing the variance of imgNum-th video frame
    vars(imgNum) = var(double(image(:)));
end

varTh = 1.5; % 1.5 suggested by Ryosuke Yamamoto <ryamamot@ks.cs.titech.ac.jp>  
% you need to play with the variacne threshold value "varTh" value to generate desirable flow
% images for your dataset, e.g set vars<=1. vars holds the variance of each video frame, e.g. vars shape is [141 x 1
% double] for UCF101 video v_Basketball_g01_c01

% filter out frames which have low variance
notkeepimg = vars<=varTh;

chnageimageinds = zeros(numFrames,1);
for imgNum = 1: length(imagelist)
    
    % if the image is in the notkeeping list 
    if notkeepimg(imgNum)
        
        % computing the mean variance of previous 2 and next 2 video frames
        meanvar = (sum(vars(max(1,imgNum-2):min(imgNum+2,numFrames)))-vars(imgNum))/4;
        
        % if the difference between the mean-variance and variance of the imgNum-th frame is
        % above varTh that means the neighbouring frames of imgNum-th frame
        % have relatively higher variance than variance of the imgNum-th frame,
        % that means intuitively, imgNum-th should have also high variance, but 
        % currently it is showing low variance, that means very less motion
        % fields, so let's replace this frame with a suitable flow frame
        if meanvar-vars(imgNum)>=varTh
            chnageimageinds(imgNum) = 1;
        end
    end
end

copyImgFrm = zeros(numFrames,1);
for imgNum = 1: length(imagelist)
    if chnageimageinds(imgNum)
        choosefrome = max(1,imgNum-2):min(imgNum+2,numFrames);
        choosefrome(choosefrome==imgNum) = []; 
        choosefrome(chnageimageinds(choosefrome)==1) = [];
        
        if ~isempty(choosefrome)
            [~, ind] = min(abs(choosefrome-imgNum));
            copyImgFrm(imgNum) = choosefrome(ind);
        end
    end
end

for imgNum = 1: length(imagelist)
    if copyImgFrm(imgNum)>0
        sourcefile = [imgdir,imagelist(copyImgFrm(imgNum)).name];
        destfile = [newimgdir,imagelist(imgNum).name];
        copyfile(sourcefile,destfile)
    else
        sourcefile = [imgdir,imagelist(imgNum).name];
        destfile = [newimgdir,imagelist(imgNum).name];
        copyfile(sourcefile,destfile);
    end
end
end
