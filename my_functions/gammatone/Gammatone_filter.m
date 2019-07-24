function [ out ] = Gammatone_filter( x,impulse_response )
% x: a signal, can be noisy/clean signal, length(signal)x1
% impulse_response: no.of filters x 800
% out: returns the filtered output using gammatone filterbank,of dimension
% no.of filters x length(signal)

r=size(impulse_response,1);
out = zeros(r,length(x));
for i=1:r
    out(i,:)=conv(x,impulse_response(i,:),'same');
end

end

