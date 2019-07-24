function [ recon ] = synthesis( W,wshift,IRM,Y_filtered,impulse_response, Alpha )

addpath(genpath('/media/ankit/Speechlab_2TB/mihir_savan_SE/SE/Toolbox/voicebox'))
addpath(genpath('/media/ankit/Speechlab_2TB/mihir_savan_SE/SE/gammatone'));
addpath(genpath('/media/ankit/Speechlab_2TB/mihir_savan_SE/SE/my_functions'));

for i=1:size(IRM,1)                   
    subband_noisy_signal = Y_filtered(i,:); 
    framed_noisy_subband = enframe(subband_noisy_signal,W,wshift);
%     size(framed_noisy_subband)
    framed_noisy_subband = framed_noisy_subband(1:length(IRM(1,:)),:);
%     size(framed_noisy_subband)
    for j=1:size(framed_noisy_subband,1)
        prod_framed_subband(j,:) = framed_noisy_subband(j,:).*IRM(i,j);
    end
    temp_signals(i,:) = overlapadd(prod_framed_subband,W,wshift);
end

% Pass subband signals through Gammatone FB
for i=1:size(temp_signals,1)
    rec_sig_filter(i,:) = conv(temp_signals(i,:), fliplr(impulse_response(i,:)),'same');
end
recon = sum(rec_sig_filter);
% recon1 = recon/max(recon);

B = [1 -Alpha];
 recon= filter(1,B,recon);
%  recon = recon/max(recon);


end

