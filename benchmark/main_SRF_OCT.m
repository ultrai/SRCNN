clearvars; close all; clc;

% ****************************************
% REQUIRED: path to Piotr's Matlab Toolbox
pmtPath = 'toolbox';
% ****************************************

addpath('helper/'); addpath(genpath('method/')); addpath(genpath(pmtPath));
%%
% if exist('data/train','dir')
%     rmdir('data/train','s')
% end
% if exist('data/test','dir')
%     rmdir('data/test','s')
% end
% if exist('models','dir')
%     rmdir('models','s')
% end
% mkdir('data/train/low/')
% mkdir('data/train/high/')
% mkdir('data/test/low/')
% mkdir('data/test/high/')
% mkdir('models')

% for Idx = 1:10
%     I = imread(['data/Images for Dictionaries and Mapping leraning/LL',num2str(Idx),'.tif']);
%     imwrite(imresize(I(:,1:2:end),[450,900]),['data/train/low//LL',num2str(Idx),'.bmp'],'bmp')
%     I = imread(['data/Images for Dictionaries and Mapping leraning/HH',num2str(Idx),'.tif']);
%     imwrite(I,['data/train/high//HH',num2str(Idx),'.bmp'],'bmp')
% end
% for Idx = 1:18
%     I = imread(['data/For synthetic experiments/',num2str(Idx),'/test.tif']);
%     imwrite(imresize(I(:,1:2:end),[450,900]),['data/test/low//LL',num2str(Idx),'.bmp'],'bmp')
%     I = imread(['data/For synthetic experiments/',num2str(Idx),'/average.tif']);
%     imwrite(I,['data/test/high//HH',num2str(Idx),'.bmp'],'bmp')
% end
%     
%% algorithm settings
sropts.datapathHigh = 'data/train/high';
sropts.datapathLow = 'data/train/low';
sropts.sf = 1;
sropts.downsample.kernel = 'bicubic';
sropts.downsample.sigma = 0;
sropts.patchSizeLow = [3 3] * sropts.sf;
sropts.patchSizeHigh = [3 3] * sropts.sf;
sropts.patchStride = [1 1] * sropts.sf;
sropts.patchBorder = [1 1] * sropts.sf;
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



%% visualize some results
fprintf('\nBicubic Upsampling (x%d): \n',sropts.sf);
psnr_m=zeros(length(outstats),1);
for i=1:length(outstats)
  psnr_m(i)=outstats(i).eval.bic.psnr;
  fprintf('Img %d/%d: psnr = %.2f dB\n',i,length(outstats),psnr_m(i));
end;
fprintf('===\nMean PSNR = %.2f dB\n', mean(psnr_m));

fprintf('\nSRF Upsampling (x%d): \n',sropts.sf);
psnr_m=zeros(length(outstats),1);
for i=1:length(outstats)
  psnr_m(i)=outstats(i).eval.srf.psnr;
  fprintf('Img %d/%d: psnr = %.2f dB\n',i,length(outstats),psnr_m(i));
end;
fprintf('===\nMean PSNR = %.2f dB\n', mean(psnr_m));

