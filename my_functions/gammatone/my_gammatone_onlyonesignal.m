function [ impulse_response,S_gtm ] = my_gammatone_onlyonesignal( clean, fs, wlen, wshift)

fl = 50; fc = 1500;
fh = fs/2;
filters_per_ERB = 2.05;

W=hamming(wlen);              % window

addpath(genpath('/home/daiict/Documents/Neil/sem3/speech/my_functions'));
addpath(genpath('/home/daiict/Documents/Neil/Toolbox/voicebox'));

analyzer = Gfb_Analyzer_new(fs, fl, fc, fh, filters_per_ERB);
bands = length(analyzer.center_frequencies_hz);
impulse = [1, zeros(1,799)];
[impulse_response, analyzer] = Gfb_Analyzer_process(analyzer, impulse);
frequency_response = fft(real(impulse_response)');
frequency = [0:799] * fs / 800;
impulse_response = real(impulse_response);

S_filtered = Gammatone_filter(clean,impulse_response);
S_gtm = energies( S_filtered,W,wshift );

end

