%this function inpaints a video

% inputs:
% 1/ input video
% 2/ occlusion video
% 3/ inpainting parameters

function[imgOut, shiftVolOut] = inpaint_video(varargin)

    imgVolIn = varargin{1};
    occVolIn = varargin{2};
    
    %FIXED PARMETERS !!!定义一些固定参数
    %GAUSSIAN PYRAMID PARAMETERS
    filterSize = 3;    %fixed 滤波器大小
    scaleStep = 0.5;    %fixed 尺度步长
    sigma = 1.5;    %fixed 标准差
    useAllPatches = 0;
    reconstructionType = 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %parse inpainting parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [maxLevel,patchSizeX,patchSizeY,patchSizeT,textureFeaturesActivated,sigmaColour,file] = ...
    parse_inpaint_parameters(varargin{3});
%调用了一个函数parse_inpaint_parameters用于解析修复参数
%并将返回的结果分配给多个变量
%计算补丁大小，并将其赋值给patchSize
    patchSize = [patchSizeX patchSizeY min(patchSizeT,size(imgVolIn,4))];
    patchMatchParams.patchSize = [patchSizeX patchSizeY min(patchSizeT,size(imgVolIn,4))];
    patchMatchParams.patchSizeX = patchSize(1);
    patchMatchParams.patchSizeY = patchSize(2);
    patchMatchParams.patchSizeT = patchSize(3);
%设置了一些与最邻近搜索和重构相关的参数
    patchMatchParams.w = max([size(imgVolIn,2) size(imgVolIn,3) size(imgVolIn,4)]);  %manual
	patchMatchParams.alpha = 0.5;   %fixed
    patchMatchParams.fullSearch = 0;
    patchMatchParams.partialComparison = 1;
    patchMatchParams.nbItersPatchMatch = 10;
    patchMatchParams.patchIndexing = 0;
    patchMatchParams.reconstructionType = reconstructionType;

    %parameters concerning the iterations of the nearest neighbour search
    %and reconstruction
    maxNbIterations = 20;%最大迭代次数
    residualThresh = 0.1;%残差阈值
    
    t1 = tic;
    
    %%%% get pyramid volumes 获取输入视频和遮挡视频的金字塔体积
    imgVolPyramid = get_image_volume_pyramid(imgVolIn,filterSize,sigma,maxLevel,size(imgVolIn,4));%patchSize(3)
    occVolPyramid = get_image_volume_pyramid(occVolIn,filterSize,sigma,maxLevel,size(imgVolIn,4));%patchSize(3)

    %%%%% create texture feature pyramid 
    %根据textureFeaturesActivated的值，决定是否计算纹理特征金字塔
    if (textureFeaturesActivated>0)
        disp('Calculating texture feature pyramids');
        featurePyramid = get_video_features(imgVolIn,occVolIn,maxLevel,file);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %    iterate over all pyramid levels 在金字塔层级上进行迭代
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ii=maxLevel:-1:1
        pp=1;
        iterationNb = 1;
        residual = inf;

        occVol = single(occVolPyramid{ii});
        
        imgVol(1,:,:,:) = imgVolPyramid{ii,1};  %get the red channel for this image volume
        imgVol(2,:,:,:) = imgVolPyramid{ii,2};  %get the blue channel for this image volume
        imgVol(3,:,:,:) = imgVolPyramid{ii,3};  %get the green channel for this image volume
        
        if (exist('featurePyramid','var'))
            gradX = single(featurePyramid{ii,1});
            gradY = single(featurePyramid{ii,2});
            normGradX = single(featurePyramid{ii,3});
            normGradY = single(featurePyramid{ii,4});

            patchMatchParams.gradX = gradX;
            patchMatchParams.gradY = gradY;
            patchMatchParams.normGradX = normGradX;
            patchMatchParams.normGradY = normGradY;
        end
        
        %if we are not at the coarsest level, then we recreate the image volume
        %using the previously upsampled shift map
        %重构图像体积
        if (ii~=maxLevel)
            occVolToInpaint = occVol;
            if (exist('featurePyramid','var') && exist('shiftVol'))
                [imgVol,gradX,gradY,normGradX,normGradY] = reconstruct_video_and_features_mex(imgVol,occVolToInpaint,...
                        shiftVol,patchMatchParams,...
                        sigmaColour,useAllPatches,reconstructionType);%
                patchMatchParams.gradX = single(gradX); patchMatchParams.gradY = single(gradY); patchMatchParams.normGradX = single(normGradX); patchMatchParams.normGradY = single(normGradY);
            else
                [imgVol] = reconstruct_video_mex(imgVol,occVolToInpaint,...
                        shiftVol,patchMatchParams,...
                        sigmaColour,useAllPatches,reconstructionType);%
            end
        end
        
        %get the finer version of the image volume
        imgVolFine(1,:,:,:) = imgVolPyramid{max(ii-1,1),1};  %get the red channel for this image volume
        imgVolFine(2,:,:,:) = imgVolPyramid{max(ii-1,1),2};  %get the blue channel for this image volume
        imgVolFine(3,:,:,:) = imgVolPyramid{max(ii-1,1),3};  %get the green channel for this image volume
        
        %create a structuring element which is the size of the patch : this
        %will be used to dilate the occlusion
        %创建一个与补丁大小相同的结构元素，用于膨胀遮挡
        structElPatch = strel('arbitrary', ones(patchSize(2),patchSize(1),patchSize(3)));
        occVolDilate = imdilate(occVol,structElPatch);

        iterationNb = iterationNb+1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % start of iterative inpainting at this level
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        while (pp<= maxNbIterations && residual > residualThresh)

            sizeImgVol = size(imgVol);
            imgVolIterMinusOne = imgVol+1-1;
            
            if (ii ~= maxLevel || pp>1)     %not bottom level
                
                patchMatchParams.partialComp = 0;
                useAllPatches = 1;
                
                if (exist('shiftVol'))
                    firstGuess = shiftVol+1-1;%zeros([size(imgVol,1) size(imgVol,2) size(imgVol,3) 4]);%
                else
                    firstGuess = single(zeros([4 size(imgVol,2) size(imgVol,3) size(imgVol,4)]));
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %carry out the 3D patchMatch
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                shiftVol = spatio_temporal_patch_match_mex( imgVol, imgVol,...
                    patchMatchParams,firstGuess,occVolDilate,occVolDilate);

                if (exist('stop_and_debug.txt'))
                    keyboard;
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %carry out the reconstruction
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if (exist('featurePyramid','var') && exist('shiftVol'))
                    %调用函数进行视频和特征的重构
                    [imgVol,gradX,gradY,normGradX,normGradY] = reconstruct_video_and_features_mex(imgVol,occVol,...
                            shiftVol,patchMatchParams,...
                            sigmaColour,useAllPatches,reconstructionType);%
                    patchMatchParams.gradX = single(gradX); patchMatchParams.gradY = single(gradY); patchMatchParams.normGradX = single(normGradX); patchMatchParams.normGradY = single(normGradY);
                else
                    %不存在特征金字塔：调用函数进行视频重构
                    [imgVol] = reconstruct_video_mex(imgVol,occVol,...
                        shiftVol,patchMatchParams,...
                            sigmaColour,useAllPatches,reconstructionType);%
                end

                iterationNb = iterationNb+1;
            else
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %      INITIALISATION       %
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %将输入的遮挡体积occVol赋给occVolIter变量
                occVolIter = occVol;
                %并计算occVolIter中所有元素的和，即遮挡点的个数
                sumHole = sum(occVolIter(:));
                
                %fill in, in an onion peel fashion 
                while(sumHole >0)
                    sumHole = sum(occVolIter(:));

                    %使用一个3x3x3的全1结构元素对occVolIter图像进行侵蚀操作
                    structElCube = strel('arbitrary', ones(3,3,3));
                    occVolErode = imerode(occVolIter,structElCube);
                    
                    %set up the partial occlusion volume for the PatchMatch :
                    % - 0 for non-occlusion;无遮挡
                    % - 1 for occluded and not to take into account when
                    % comparing patches不能考虑匹配的遮挡部分
                    % - 2 for occluded and to take into account (we do not allow
                    %the nearest neighbours to point to these pixels, but
                    %they have been reconstructed at this
                    %iteration已经修复可以参与匹配的遮挡
                    occVolPatchMatch = occVolDilate;
                    occVolPatchMatch((occVolDilate - occVolIter) == 1) = 2;
                    
                    %initial guess : by default, set everything to 0初始化高斯核
                    if (~exist('shiftVol'))
                        firstGuess = single(zeros([4 size(imgVol,2) size(imgVol,3) size(imgVol,4)]));
                    else
                        firstGuess = shiftVol+1-1;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %carry out the 3D patchMatch
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %在当前层级上进行3D PatchMatch匹配
                    shiftVol = spatio_temporal_patch_match_mex(imgVol, imgVol,...
                     patchMatchParams,firstGuess,occVolPatchMatch,occVolDilate);

                    if (exist('stop_and_debug.txt'))
                        keyboard;
                    end
                    
                    %determine the pixels to reconstruct at this layer
                    occVolBorder = abs(occVolIter - occVolErode);
                    %if the occVol == 2, then we cannot use the colour
                    %info, but we do not inpaint it either
                    occVolBorder(occVolErode == 1) = 2;
                    %【？】得到的 occVolBorder 就表示在该层级上需要进行修复的像素点
                    
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %carry out the reconstruction
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if (exist('featurePyramid','var') && exist('shiftVol'))
                        %调用函数对图像体进行重建
                        [imgVol,gradX,gradY,normGradX,normGradY] = reconstruct_video_and_features_mex(imgVol,occVolBorder,...
                            shiftVol,patchMatchParams,...
                            sigmaColour,useAllPatches,reconstructionType);%
                        patchMatchParams.gradX = single(gradX); patchMatchParams.gradY = single(gradY); patchMatchParams.normGradX = single(normGradX); patchMatchParams.normGradY = single(normGradY);
                    else
                        [imgVol] = reconstruct_video_mex(imgVol,occVolBorder,...
                            shiftVol,patchMatchParams,...
                                sigmaColour,useAllPatches,reconstructionType);%
                    end
                    
                    %now go to the next onion peel layer
                    occVolIter = occVolErode;
                    %add 1 to iteration number
                    iterationNb = iterationNb+1;
                end
            end
            pp=pp+1;
            %we have finished the initialisation : make sure that both
            %the patchMatch AND the reconstruction consider that
            %everything is known (no more partial patch comparisons)
            patchMatchParams.partialComparison = 0;
            useAllPatches = 1;  patchMatchParams.useAllPatches = 1;

            %calculate the residual to see if we terminate
            residual = sum(abs(imgVolIterMinusOne(:) - imgVol(:)))/(single(3*sum(occVol(:)>0)))
        end
        
        beep;

        if (ii>1)
            imgVol = imgVolFine;%[];%
            %interpolate the shift volume
            shiftVol = single(interpolate_disp_field(shiftVol,imgVol,1/scaleStep, patchSize,'nearest'));
        end

        if (ii==1)
            t2 = toc(t1)
            imgVol = reconstruct_video_mex(imgVol,occVol,...
                        shiftVol,patchMatchParams,sigmaColour,useAllPatches,1);
            occInds = find(occVol>0);
            energy = sum(shiftVol(occInds + 3*prod(sizeImgVol(1:2))));
            imgOut = imgVol;
            shiftVolOut = shiftVol;
            
            return;
        end
        
        %erase some of the video structures
        occVolDilate = single([]);
        occVolPatchMatch = single([]);
        imgVolFine = single([]);

        patchMatchParams.partialComp = 0;
        useAllPatches = 1;

    end
    imgOut = imgVol;
end

