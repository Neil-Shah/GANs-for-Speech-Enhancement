function [ energy_filtered ] = energies( filtered_signal,W,wshift )

energy_filtered=[];
[r,c] = size(filtered_signal);
for i=1:r
    subband_signal=filtered_signal(i,:);
    framed_subband_signal=enframe(subband_signal,W,wshift);
    energy=sqrt(mean(framed_subband_signal'.^2));
    energy_filtered=[energy_filtered; energy];
end

end

