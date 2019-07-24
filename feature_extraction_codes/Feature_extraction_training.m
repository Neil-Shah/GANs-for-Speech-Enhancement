clc; clear; close all;

% add the path for my_functions
addpath(genpath('/home/Documents/my_functions'));

% define parameters for Gammatonegram features extraction
wshift = 80;                   % 5 ms
wlen = 400;                    % DFT window length
W=hamming(wlen);               % window
beta = 1;
lftcontxt = 2; rhtcontxt = 2;

% define the path where the clean and noisy wave files are present
clean_path = '/home/Documents/train/Clean/';
noisy_path = '/home/Documents/train/Noisy/';

% create the directories and define the path where the extracted features need to be stored
mkdir /home/Documents batches
mkdir /home/Documents/batches Training_complementary_feats
files_clean = dir([clean_path,'/*.wav']);
files_noisy = dir([noisy_path,'/*.wav']);
save_path = '/home/Documents/batches/Training_complementary_feats/';

% define the batch size for neural network training
batch_size = 1000;
batch_ind = 0;
frames = 0;

% initialize the input and output with some rough estimate
IP_feats = zeros(250000,(lftcontxt+rhtcontxt+1)*64); % input noisy context
OP_clean_spec = zeros(250000,64); % output training labels

% taking first 350 files for training (user specific)
for i = 1:350
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
    IP_feats(frames+1:frames+Min_frame,:) = GT_feat';
    OP_clean_spec(frames+1:frames+Min_frame,:) = Clean_gtm';  
    
    frames = frames+Min_frame;
end

IP_feats = IP_feats(1:frames,:);
OP_clean_spec = OP_clean_spec(1:frames,:);

Feat = zeros(batch_size,(lftcontxt+rhtcontxt+1)*64);
Clean_cent = zeros(batch_size,64);

k=0;
Rand_frames = randperm(frames);
while k*batch_size+batch_size <= frames
    cur_frames = Rand_frames(k*batch_size+1:k*batch_size+batch_size);
    
    Feat = single(IP_feats(cur_frames,:));
    Clean_cent = single(OP_clean_spec(cur_frames,:));
    k=k+1;
    save([save_path,'Batch_',num2str(batch_ind)],'Feat','Clean_cent');
    
    batch_ind = batch_ind+1;
end
