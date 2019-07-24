clc; clear; close all;

% add the path for my_functions
addpath(genpath('/home/Documents/my_functions'));

% define parameters for Gammatonegram features extraction
wshift = 80;                   % 5 ms
wlen = 400;                    % 28 ms,window length
W=hamming(wlen);               % window
beta = 1;
lftcontxt = 2; rhtcontxt = 2;

% define the path where the clean and noisy wave files are present
clean_path = '/home/Documents/train/Clean/';
noisy_path = '/home/Documents/train/Noisy/';

% create the directories and define the path where the extracted features need to be stored
mkdir /home/Documents/batches Validation_complementary_feats
files_clean = dir([clean_path,'/*.wav']);
files_noisy = dir([noisy_path,'/*.wav']);
save_path = '/home/Documents/batches/Validation_complementary_feats/';

% define the batch size for neural network training
batch_size = 1000;
batch_ind = 0;
frames = 0;
k=0;

% taking next few files for validation
for i=351:400
    disp(['Processing file : ', num2str(i)])
    clean_file =[clean_path,files_clean(i).name];
    noisy_file = [noisy_path,files_noisy(i).name];
    [clean,fs] = audioread(clean_file);
    noisy = audioread(noisy_file);

    % extract feature
    [ ~,Clean_gtm,Noisy_gtm,~,~ ] = my_gammatone( clean, noisy, fs, wlen, wshift,0.95);
    Min_frame = length(Clean_gtm(1,:));
    
    % log and MVN to network's input features
    Log_Noisy_gtm = log(Noisy_gtm);
    Log_Noisy_gtm = (Log_Noisy_gtm-mean(Log_Noisy_gtm(:)))/std(Log_Noisy_gtm(:));
    GT_feat = framecontext(Log_Noisy_gtm, length(Log_Noisy_gtm(:,1)),lftcontxt,rhtcontxt); 
    
    % input and output features concatenated
    IP_feats = GT_feat';
    OP_clean_spec = Clean_gtm';

    Feat = single(IP_feats);
    Clean_cent = single(OP_clean_spec);

    save([save_path,'Test_Batch_',num2str(k)],'Feat','Clean_cent');
    k=k+1;
    
end
