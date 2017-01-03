%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  The following is a demo that can be used to run the Sparsity-based SBSDI
%  image reconstruction code
%
%  If you are using this code to generate results, please cite the
%  following papers:
%  Leyuan Fang, Shutao Li, Ryan P. McNabb?, Qing Nie, Anthony N. Kuo,
%  Cynthia A. Toth, Joseph A. Izatt, and Sina Farsiu, "Fast Acquisition and
%  Reconstruction of Optical Coherence Tomography Images via Sparse
%  Representation" IEEE Transactions on Medical Imaging, In press, 2013
%  Leyuan Fang
%  Department of Ophthalmology,
%  Duke University Medical Center,Durham, NC, 27710, USA
%  fangleyuan@gmail.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;
win_w = 4;  % The width of the patch used for the subsampled image
win_h = 4;  % The length of the patch used for the subsampled image
step = 1;   % The sliding distance between each extracted patches
scale_factor =4; % Downsampling factor
sparsity = 3; % The stopping condition for COMP algorithm
Beta = 80;  % The parameter for computing the weights
% set the paramters for the OMP_mex
param.L=sparsity; % not more than 10 non-zeros coefficients
param.eps=0.1; % squared norm of the residual should be less than 0.1
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
% and uses all the cores of the machine
PSNR_total =[];
I_Pred_chap4=[];
I_GT_chap4=[];
CurrentPatch = ([pwd,'\For synthetic experiments\']); %please input the path of fold  “For synthetic experiments”  and add \.
for I_Num = 1:18
    if I_Num~=9
    % Load the High-SNR-High-Resoluion averaged image if it is available in the
    % Synthetic Experiments
    num = num2str(I_Num);
    Testpath = strcat(CurrentPatch,num);
    Trupath = ('\average.tif');
    Truthpath =  strcat(Testpath,Trupath);
    Truth = double(imread(Truthpath));
    
    % Load Low-SNR-low-resolution Test image and its nearby slices
    Test = {};
    T1path = ('\test.tif');
    Test1path =  strcat(Testpath,T1path);
    LTest = double(imread(Test1path)); % load the test image
    % Test{1} = LTest; % For real subsampled image experiment
    Test{1} = LTest(:,1:scale_factor:end); % For synthetic image
    % experiment, image needs to be further downsampled
    
    Test{2} = Test{1};
    
    Test{3} = Test{1};
    
    Test{4} = Test{1};
    
    Test{5} = Test{1};
    
    % Load more nearby slices to process simultaneously
    % LTest = double(imread('5.tif'));
    % Test{6} = LTest;
    % Test{6} = LTest(:,1:scale_factor:end);
    %
    % LTest = double(imread('6.tif'));
    % Test{7} = LTest;
    % Test{7} = LTest(:,1:scale_factor:end);
    
    % LTest = double(imread('7.tif'));
    % Test{8} = LTest;
    % Test{8} = LTest(:,1:scale_factor:end);
    
    % Extract patches from the current processed test image, its nearby nearby slices and their high-pass images
    Testp = {};  % preserve the patches from teh current processed test image and its nearby slices in Testp
    HF_Testp = {}; % preserve the high-frequency patches from the current processed high-pass test image and its nearby high-pass slices in HF_Testp
    sigma     =   2.4;  % the standard deviation for the Gaussian lowpass filter
    psf       =   fspecial('gauss', [win_w win_h], sigma); % point spread function for blurring the training image
    % tic
    for ii =  1: size(Test,2)
        [HF_Testp{ii}, Testp{ii}    ]       =   Get_patches(Test{ii}, win_w,win_h, psf, step,step,1,scale_factor, 0);
    end
    % toc
    % Divde the patches in the curent processed image into two groups: smooth and details
    nsig = function_stdEst(Test{1});
    delta       =   sqrt(nsig^2+16);
    v           =   sqrt( mean( HF_Testp{1}.^2 ) );
    [a, i0]     =   find( v<delta );
    set         =   1:size(HF_Testp{1}, 2);
    set(i0)     =   []; % i0: index set for detailed patches, set: index set for smooth patches
    
    
    %%% Load Dictionaries, its correponding centroids and Mapping functions
    load HD_det_4times; % load high-SNR-high-resolution structral dictionary for detailed patches
    load LD_det_4times; % load low-SNR-low-resolution structral dictionary for detailed patches
    load LCent_det_4times; % load centroids for detailed group
    load HD_smoot_4times; % load high-SNR-high-resolution structral dictionary for smooth patches
    load LD_smoot_4times; % load low-SNR-low-resolution structral dictionary for smooth patches
    load LCent_smoot_4times; % load centroids for smooth group
    load Map_det_4times.mat  % mapping function for detailed patches
    load Map_smooth_4times.mat  % mapping function for smooth group
    
    
    
    L_patch_size = win_w*win_h;
    H_patch_size = win_w*win_h*scale_factor;
    for ii = 1: size(HD_det,2)
        HD_det{ii} = HD_det{ii} - repmat(mean(HD_det{ii}), [H_patch_size 1]);
        HD_det{ii} = HD_det{ii}./repmat(sqrt(sum(HD_det{ii}.^2, 1)),H_patch_size, 1);
        LD_det{ii} = LD_det{ii} - repmat(mean(LD_det{ii}), [L_patch_size 1]);
        LD_det{ii} = LD_det{ii}./repmat(sqrt(sum(LD_det{ii}.^2, 1)),L_patch_size, 1);
    end
    
    for ii = 1: size(HD_smoot,2)
        HD_smoot{ii} = HD_smoot{ii} - repmat(mean(HD_smoot{ii}), [H_patch_size 1]);
        HD_smoot{ii} = HD_smoot{ii}./repmat(sqrt(sum(HD_smoot{ii}.^2, 1)),H_patch_size, 1);
        LD_smoot{ii} = LD_smoot{ii} - repmat(mean(LD_smoot{ii}), [L_patch_size 1]);
        LD_smoot{ii} = LD_smoot{ii}./repmat(sqrt(sum(LD_smoot{ii}.^2, 1)),L_patch_size, 1);
    end
    % Assign each detaled patch to one Detailed Subdictionary
    L1           =   size(set,2);
    cls_idx1     =   zeros(size(Testp{1},2) , 1);
    
    vec = LCent_det';
    for j = 1 : L1
        dis   =   vec(:, 1) -  Testp{1}(1, set(j));
        for i = 2 : win_w*win_h
            dis  =  dis + (vec(:, i)-Testp{1}(i, set(j))).^2;
        end
        [val ind]      =   min( dis );
        cls_idx1( set(j) )   =   ind;
    end
    
    [s_idx1, seg1]   =  Proc_cls_idx( cls_idx1 );
    
    % Assign each smooth patch to one smooth Subdictionary
    L2           =   size(i0,2);
    cls_idx2     =   zeros(size(Testp{1},2) , 1);
    
    vec = LCent_smoot';
    for j = 1 : L2
        dis   =   vec(:, 1) -  Testp{1}(1, i0(j));
        for i = 2 : win_w*win_h
            dis  =  dis + (vec(:, i)-Testp{1}(i, i0(j))).^2;
        end
        [val ind]      =   min( dis );
        cls_idx2( i0(j) )   =   ind;
    end
    
    [s_idx2, seg2]   =  Proc_cls_idx( cls_idx2 );
    
    % Compute the weights for patches from nearby slices
    hp        =  max(Beta*nsig, 80);
    for jj = 1: size(Testp,2)
        dist_mean = mean(abs(Testp{jj} - Testp{1}).^2);
        Dis(jj,:) = dist_mean;
    end
    wei_arr_temp = exp( -Dis./hp );
    wei_arr_temp_sum = sum(wei_arr_temp,1);
    wei_arr = wei_arr_temp./ (repmat(wei_arr_temp_sum, [size(Dis,1) 1])+eps);
    
    
    % To accurately estimate the mean of the HH patches, need to reduce the
    % noise in the LL patches
    Fileted_patches = patchfiltering (Testp,wei_arr,win_w, win_h,L_patch_size,Test{1});
    
    % Sparse representation of detailed patches
    [mm , nn] = size(Testp{1});
    YY = zeros(mm*scale_factor,nn);
    nblk = size(Test,2);
    
    % tic
    for ii = 1: size(seg1,1) -2
        dex = seg1(ii+1)+1: seg1(ii+2);
        Index = s_idx1(dex);
        Dct_index = cls_idx1(s_idx1(seg1(ii+1)+1)); % selected structral subdictionary
        
        % Current processed patch and the patches from nearby slices
        % substract the mean from the proceesed patches
        mean_L = {};
        mean_H = {};
        ZZ = zeros(win_w*win_h, size(Index,1)*nblk) ;
        for iii = 1: nblk
            mean_L{iii} = repmat(mean(Testp{iii}(:,Index)), [L_patch_size 1]); % mean of the LL pathes
            mean_H = mean(Fileted_patches(:,Index));% mean of the HH patches, using the filtered LL patches to eastimate
            Testp{iii}(:,Index) = Testp{iii}(:,Index) - mean_L{iii}; % substract the mean in the test LL patches
            ZZ(:,iii:nblk: end - (nblk-iii))  =  Testp{iii}(:,Index); % combine the patches of the nearby slices into one matrix
        end
        
        % simultaneouly sparse representation patches of nearby slices
        ind_groups=int32(0:nblk:size(ZZ,2)-1); % indices of the first signals in each group
        if nblk == 1 % for the single slice condition
            Coefs=mexOMP(ZZ,LD_det{Dct_index},param);
        else % for multiple slices condtion
            Coefs=mexSOMP(ZZ,LD_det{Dct_index},ind_groups,param);
        end
        Coef_tem = Map_det{Dct_index}*Coefs;
        %     Coef_tem =Coefs; % without using the map
        Aver_patches = HD_det{Dct_index}*Coef_tem;   % recovered high resolution patches in nearby slices
        %     tic
        jk = 1;
        for kkk = 0 :nblk:size(ZZ,2) -nblk
            v = wei_arr( :,Index(jk));
            Aver_patches(:,kkk + 1:kkk+nblk) = Aver_patches(:,kkk + 1:kkk+nblk)+ mean_H(jk);
            YY(:, Index(jk)) = Aver_patches(:,kkk + 1:kkk+nblk)*v; % recovered high resolution patches in the current processed image with the weights averaging
            jk = jk +1;
        end
        
    end
    
    % Sparse representation of smooth patches
    
    for ii = 1: size(seg2,1) -2
        dex = seg2(ii+1)+1: seg2(ii+2);
        Index = s_idx2(dex);
        Dct_index = cls_idx2(s_idx2(seg2(ii+1)+1));
        
        ZZ = zeros(win_w*win_h, size(Index,1)*nblk) ;
        mean_L = {};
        mean_H = {};
        for iii = 1: nblk
            mean_L{iii} = repmat(mean(Testp{iii}(:,Index)), [L_patch_size 1]); % mean of the LL pathes
            mean_H = mean(Fileted_patches(:,Index));% mean of the HH patches, using the filtered LL patches to eastimate
            Testp{iii}(:,Index) = Testp{iii}(:,Index) - mean_L{iii}; % substract the mean in the test LL patches
            ZZ(:,iii:nblk: end - (nblk-iii))  =  Testp{iii}(:,Index);  % combine the patches of the nearby slices into one matrix
        end
        
        % simultaneouly sparse representation patches of nearby slices
        ind_groups=int32(0:5:size(ZZ,2)-1); % indices of the first signals in each group
        if nblk == 1
            Coefs=mexOMP(ZZ,LD_smoot{Dct_index},param);
        else
            Coefs=mexSOMP(ZZ,LD_smoot{Dct_index},ind_groups,param);
        end
        %     Coef_tem =Coefs;
        Coef_tem = Map_smooth{Dct_index}*Coefs;
        Aver_patches = HD_smoot{Dct_index}*Coef_tem;
        %     tic
        jk = 1;
        for kkk = 0 :nblk:size(ZZ,2) -nblk
            v = wei_arr( :,Index(jk));
            Aver_patches(:,kkk + 1:kkk+nblk) = Aver_patches(:,kkk + 1:kkk+nblk)+ mean_H(jk);
            YY(:, Index(jk)) = Aver_patches(:,kkk + 1:kkk+nblk)*v; % recovered high resolution patches in the current processed image
            jk = jk +1;
        end
        
    end
    
    % toc
    %%%Reconstruction
    [hh, ww] = size(Test{1});
    im_out   =  zeros(hh,ww*scale_factor);
    im_wei   =  zeros(hh,ww*scale_factor);
    
    N     =  hh-win_w+1;
    M     =  ww-win_h+1;
    MM = ww*scale_factor-win_h*scale_factor+1;
    % L     =  N*M;
    r     =  [1:step:N];
    r     =  [r r(end)+1:N];
    c     =  [1:step*scale_factor:MM];
    c     =  [c c(end)+1:MM];
    % X     =  zeros(b*b,L,'single');
    
    % weighted average of overlapped patches for image reconstruction
    k        =  0;
    for i  = 1:win_w
        for j  = 1:win_h*scale_factor
            k    =  k+1;
            im_out(r-1+i,c-1+j)  =  im_out(r-1+i,c-1+j) + reshape( YY(k,:)', [N M]);
            im_wei(r-1+i,c-1+j)  =  im_wei(r-1+i,c-1+j) + 1;
        end
    end
    im_out    =  im_out./(im_wei+eps);
    PSNR_result            =   PSNR( double(im_out), Truth, 0, 0 ) % If the
    % averaged image is avaiblable, compute the PSNR between the recovered
    % image and averaged image.
    I_Pred_chap4 = cat(3,I_Pred_chap4,double(im_out));
    I_GT_chap4 = cat(3,I_GT_chap4,Truth);
    
    PSNR_total = [PSNR_total;PSNR_result];
    % figure; imshow(im_out,[]);
    Rpath = ('Synthetic_Reconstructed_result.tif');
    R1path = strcat(num, Rpath);
    outputpath = strcat(CurrentPatch, R1path);
    imwrite(uint8(im_out),outputpath)
end
end
mean(PSNR_total)
I_Pred_chap4_t = [];
I_GT_chap4_t = [];

for temp = 1:17
    I_Pred_chap4_t = [I_Pred_chap4_t;I_Pred_chap4(:,:,temp)];
    I_GT_chap4_t = [I_GT_chap4_t;I_GT_chap4(:,:,temp)];
    ssim(I_Pred_chap4(:,:,temp)/255,I_GT_chap4(:,:,temp)/255)
end
PSNR(I_Pred_chap4_t, I_GT_chap4_t, 0, 0 ) 