function a = gpu(x)
% nickname of gpuArray and make x into single accuracy to speed up
a = gpuArray(single(x));


end % end of the file