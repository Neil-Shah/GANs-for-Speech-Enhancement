clc; clear; close all;

% add the path for my_functions
addpath(genpath('/home/Documents/my_functions'));

% taking 21 files for testing
number = 21;

% define parameters for Gammatonegram features extraction
wshift = 80;                   % 5 ms
wlen = 400;                    % 28 ms,window length
W=hamming(wlen);               % window
beta = 1;
lftcontxt = 2; rhtcontxt = 2;

% define the path where the clean and noisy wave files are present
clean_path = '/home/Documents/test/Clean/';
noisy_path = '/home/Documents/test/Noisy/';

% create the directories and define the path where the extracted features need to be stored
%mkdir /home/Documents batches
mkdir /home/Documents/batches Testing_complementary_feats
files_clean = dir([clean_path,'/*.wav']);
files_noisy = dir([noisy_path,'/*.wav']);
save_path = '/home/Documents/batches/Testing_complementary_feats/';

% define the batch size for neural network training
batch_size = 1000;
batch_ind = 0;
frames = 0;

k=0;
for i=1:number
    disp(['Processing file : ', num2str(i)])
    clean_file =[clean_path,files_clean(i).name];
    noisy_file = [noisy_path,files_noisy(i).name];
    [clean,fs] = audioread(clean_file);
    noisy = audioread(noisy_file);
    
    % extract feature
    [ ~,Clean_gtm,Noisy_gtm,~,~ ] = my_gammatone( clean, noisy, fs, wlen, wshift,0.95);
    
    % log and MVN to network's input features
    Log_Noisy_gtm = log(Noisy_gtm);
    Log_Noisy_gtm = (Log_Noisy_gtm-mean(Log_Noisy_gtm(:)))/std(Log_Noisy_gtm(:));
    GT_feat = framecontext(Log_Noisy_gtm, length(Log_Noisy_gtm(:,1)),lftcontxt,rhtcontxt); 

    % input and output features concatenated
    IP_feats = GT_feat';
    OP_clean_spec = Clean_gtm';
    
    Feat = IP_feats;
    Clean_cent = OP_clean_spec;

    save([save_path,'Test_Batch_',num2str(i-1)],'Feat','Clean_cent','-v6');
    k=k+1;
    
end
