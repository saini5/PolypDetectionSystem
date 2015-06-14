
workingDir = 'C:\Users\aman_mclarenf1\Desktop\Capstone\polyp\TrainingSet_NewGTPart1of3\ShortVD_np_5\';
%mkdir(workingDir)
mkdir(workingDir,'images')
cd C:\Users\aman_mclarenf1\Desktop\Capstone\polyp\TrainingSet_NewGTPart1of3\ShortVD_np_5\;
noPolypVideo = VideoReader('nopolyp.wmv');

cd C:\Users\aman_mclarenf1\Desktop\Capstone\polyp;
xyloObj = VideoReader('drift.mp4');

nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;

% Preallocate movie structure.
mov(1:nFrames) = ...
    struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
           'colormap', []);

% Read one frame at a time.
for k = 1 : nFrames
    mov(k).cdata = read(xyloObj, k);
end

% Size a figure based on the video's width and height.
hf = figure;
set(hf, 'position', [150 150 vidWidth vidHeight])

% Play back the movie once at the video's frame rate.
movie(hf, mov, 1, xyloObj.FrameRate);

% ii = 1;
% totalNumber = noPolypVideo.NumberOfFrames
% numFrames = 1
% while numFrames>0
%    img = read(noPolypVideo,1);
%    imshow(img);
%    pause;
%    filename = [sprintf('%03d',ii) '.tif'];
%    fullname = fullfile(workingDir,'images',filename);
%    imwrite(img,fullname)    % Write out to a tif file img1.tif, img2.tif, etc
%    ii = ii+1;
%    numFrames = numFrames - 1;
% end
% %vid = video which needs to be extracted
% 
% vid = 'nopolyp.wmv';
% 
% readerobj = VideoReader(vid);
% 
% vidFrames = read(readerobj);
% 
% %get number of frames
% 
% numFrames = get(readerobj, 'numberOfFrames');
% 
% for k = 1 : numFrames
% 
%     mov(k).cdata = vidFrames(:,:,:,k);
% 
%     mov(k).colormap = [];
% 
%     %imshow(mov(k).cdata);
% 
%     imagename=strcat(int2str(k), '.jpg');
% 
%     %save inside output folder
% 
%     imwrite(mov(k).cdata, strcat('outputframe-',imagename));
% 
% end
% 
% % 
% % 
% % ii = 1;
% % 
% % totalNumber = noPolypVideo.NumberOfFrames
% % numFrames = noPolypVideo.NumberOfFrames
% % while numFrames>0
% %    j = totalNumber - numFrames + 1
% %    images = read(noPolypVideo,j);
% %    %imshow(images)
% %    pause;
% % %    filename = [sprintf('%03d',ii) '.tif'];
% % %    fullname = fullfile(workingDir,'images',filename);
% % %    imwrite(img,fullname)    % Write out to a tif file img1.tif, img2.tif, etc
% % %    ii = ii+1;
% %    numFrames = 0;