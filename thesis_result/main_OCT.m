clearvars; close all; clc;

% ****************************************
% REQUIRED: path to Piotr's Matlab Toolbox
pmtPath = 'toolbox';
% ****************************************

addpath('helper/'); addpath(genpath('method/')); addpath(genpath(pmtPath));
%%
if exist('data/train','dir')
    rmdir('data/train','s')
end
if exist('data/test','dir')
    rmdir('data/test','s')
end
if exist('models','dir')
     rmdir('models','s')
end
mkdir('data/train/low/')
mkdir('data/train/high/')
mkdir('data/test/low/')
mkdir('data/test/high/')
mkdir('models')

for Idx = 1:10
    I = imread(['data/Images for Dictionaries and Mapping leraning/LL',num2str(Idx),'.tif']);
    imwrite(imresize(I(:,1:2:end),[450,900]),['data/train/low//LL',num2str(Idx),'.bmp'],'bmp')
    I = imread(['data/Images for Dictionaries and Mapping leraning/HH',num2str(Idx),'.tif']);
    imwrite(I,['data/train/high//HH',num2str(Idx),'.bmp'],'bmp')
end
for Idx = 1:18
    if Idx~=9
        I = imread(['data/For synthetic experiments/',num2str(Idx),'/test.tif']);
        imwrite(imresize(I(:,1:2:end),[450,900]),['data/test/low//LL',num2str(Idx),'.bmp'],'bmp')
        I = imread(['data/For synthetic experiments/',num2str(Idx),'/average.tif']);
        imwrite(I,['data/test/high//HH',num2str(Idx),'.bmp'],'bmp')
    end
end
    
%% algorithm settings
sropts.datapathHigh = 'data/train/high';
sropts.datapathLow = 'data/train/low';
sropts.sf = 1;
sropts.downsample.kernel = 'bicubic';
sropts.downsample.sigma = 0;
sf = 3;
sropts.patchSizeLow = [3 3] * sf;
sropts.patchSizeHigh = [3 3] * sf;
sropts.patchStride = [1 1] ;
sropts.patchBorder = [1 1] ;
sropts.nTrainPatches = 0;
sropts.nAddBaseScales = 0;
sropts.patchfeats.type = 'filters';
O = zeros(1, sropts.sf-1);
G = [1 O -1]; % Gradient
L = [1 O -2 O 1]/2; % Laplacian
sropts.patchfeats.filters = {G, G.', L, L.'}; % 2D versions
sropts.interpkernel = 'bicubic';
sropts.pRegrForest = forestRegrTrain();
sropts.pRegrForest.M = 10;
sropts.pRegrForest.maxDepth = 15;
sropts.pRegrForest.nodesubsample = 512;
sropts.pRegrForest.verbose = 1;
sropts.pRegrForest.usepf = 1; % matlabpool open required!
sropts.useARF = 0; % requires longer training times!

% path to the model file
srforestPath = 'models';
srforestFNm = sprintf('srf_sf-%d_T-%02d_ARF-%d.mat',sropts.sf,...
  sropts.pRegrForest.M,sropts.useARF);
srforestFNm = fullfile(srforestPath,srforestFNm);

% path to test images
datapathTestHigh = 'data/test/high';
datapathTestLow = 'data/test/low';


%% train the super-resolution forest
if ~exist(srforestFNm,'file')
  fprintf('Training super-resolution forest\n');
  srforest = srForestTrain(sropts);
  srForestSave(srforestFNm,srforest);
else
  fprintf('Loading super-resolution forest\n');
  srforest = srForestLoad(srforestFNm);
end



%% testing the learned model
outstats = srForestApply(datapathTestLow,datapathTestHigh,...
  srforest,{'rmborder',3});

I_Pred_chap4_t = [];
I_GT_chap4_t = [];

for temp = 1:17
    I_Pred_chap4_t = [I_Pred_chap4_t;outstats(temp).im];
    I_GT_chap4_t = [I_GT_chap4_t;outstats(temp).GT];
    
    
end
save('SRF.mat','outstats')
psnr(I_Pred_chap4_t, I_GT_chap4_t) 
ssim(double(I_Pred_chap4_t), double(I_GT_chap4_t))
PSNR_total = zeros(17,1);
SSIM_total = zeros(17,1);
MSE_total = zeros(17,1);

MAXERR_total = zeros(17,1);
L2RAT_total = zeros(17,1);
for Idx = 1:17
    I_Pred = outstats(Idx).im;
    I_GT = outstats(Idx).GT;
%     if Idx<9
%         I_GT = imread([datapathTestHigh,'/HH',num2str(Idx),'.bmp']);
%         I_GT = imageTransformColor(I_GT);
%     else
%         I_GT = imread([datapathTestHigh,'/HH',num2str(Idx+1),'.bmp']);
%          I_GT = imageTransformColor(I_GT);
%     end
    %figure,imshow([I_GT,I_Pred])
    
%     PSNR_total =  cat(1,PSNR_total,psnr(I_GT,I_Pred));
%     SSIM_total =  cat(1,SSIM_total,ssim(double(I_Pred),double(I_GT)));
%     MSE_total =  cat(1,MSE_total,immse(double(I_Pred)*255,double(I_GT)*255));
    [PSNR_total(Idx),MSE_total(Idx),MAXERR_total(Idx),L2RAT_total(Idx)]= measerr(double(I_GT)*255,double(I_Pred)*255);
   SSIM_total(Idx) =  ssim(double(I_Pred),double(I_GT));
    
end
mean(MSE_total)
res = [PSNR_total MSE_total MAXERR_total L2RAT_total SSIM_total];
mean(res)
std(res)

%res =  [26.2039182315040,155.843444200712,209.000002741814,0.973378103685087,0.611189441708783;28.4460451702311,92.9984786070923,178.000004589558,1.00401476648029,0.659033588790796;25.2437981529590,194.401918063890,234.000001251698,0.888302977418118,0.614837442151940;22.9211569506280,331.867455721169,164.000005424023,0.850526865256500,0.590628625770846;26.3819740588459,149.583246577577,198.000003397465,1.04711068242025,0.614144963978616;27.2215464571869,123.289393214968,207.000002861023,0.961643529417586,0.636706183196205;26.9376696043544,131.617428195380,197.000003457069,0.954806125486404,0.633916448802364;25.0377943503446,203.845404823545,204.000003039837,0.982641528267761,0.578492947480588;23.3942014984698,297.618697030646,201.000003218651,1.05258646445164,0.580825134378705;28.0780457185373,101.222203921467,181.000004410744,1.01764010576474,0.636891138737512;27.9352406370781,104.605928129501,192.000003755093,1.02548532616554,0.648471233579014;23.6460931693451,280.847811459673,179.000004529953,1.22058140150471,0.612757343126566;27.1332373116672,125.822013176807,219.000002145767,1.01857800223906,0.622836881274833;26.3407047379272,151.011453278650,190.000003874302,0.967984602452878,0.611031686373267;27.5172116912302,115.175257462028,192.000003755093,0.986803406102163,0.639829523386628;25.7726742273748,172.112708134371,222.000001966953,0.952046390902511,0.620646895322097;26.7848613048378,136.330877596602,211.000002622604,0.919245105916662,0.624395597812706]

%% visualize some results
% fprintf('\nBicubic Upsampling (x%d): \n',sropts.sf);
% psnr_m=zeros(length(outstats),1);
% for i=1:length(outstats)
%   psnr_m(i)=outstats(i).eval.bic.psnr;
%   fprintf('Img %d/%d: psnr = %.2f dB\n',i,length(outstats),psnr_m(i));
% end;
% fprintf('===\nMean PSNR = %.2f dB\n', mean(psnr_m));
% 
% fprintf('\nSRF Upsampling (x%d): \n',sropts.sf);
% psnr_m=zeros(length(outstats),1);
% for i=1:length(outstats)
%   psnr_m(i)=outstats(i).eval.srf.psnr;
%   fprintf('Img %d/%d: psnr = %.2f dB\n',i,length(outstats),psnr_m(i));
% end;
% fprintf('===\nMean PSNR = %.2f dB\n', mean(psnr_m));

