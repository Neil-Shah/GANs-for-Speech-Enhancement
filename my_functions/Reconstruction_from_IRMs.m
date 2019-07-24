clc; clear; close all;
warning off;

%% Initialization
addpath(genpath('/home/Documents/my_functions'));
addpath(genpath('/home/Documents/voicebox'));
addpath(genpath('/home/Documents/my_functions/objective_measures/quality'));

patch_size = 64;
number = 824;
OV=2;                         % overlap factor of 2 (4 is also often used)
wshift=160;                   % set frame increment in samples
wlen=wshift*OV;               % DFT window length
W=hamming(wlen);              % window
N_fft = 256;                  % n-point FFT
beta = 1;

%% path of actual clean and noisy database
clean_path = '/home/Documents/train/Clean/';
noisy_path = '/home/Documents/train/Noisy/';
file_clean = dir([clean_path,'*.wav']);
file_noisy = dir([noisy_path,'*.wav']);

%% path of stored mask, after getting predicted masks
pathpredicted_mask = '/home/Documents/GAN/';
list_mask = dir([pathpredicted_mask,'*.mat']);

%% make different directories
mkdir /home/Documents Waveforms_model11
mkdir /home/Documents/Waveforms_model11 Ganrmse
mkdir /home/Documents/Waveforms_model11 Oracle
mkdir /home/Documents/Waveforms_model11 clean
mkdir /home/Documents/Waveforms_model11 noisy

%% path to store clean,noisy,oracle,enhanced waveforms
clean_wavform = '/home/Documents/Waveforms_model11/clean/';
noisy_wavform = '/home/Documents/Waveforms_model11noisy/';
gan_wavform = '/home/Documents/Waveforms_model11Ganrmse/';
actual_wavform = '/home/Documents/Waveforms_model11Oracle/';

%% specify path of enhanced waveforms
path_orac_enhanced = '/home/Documents/Waveforms_model11/';
wheretostore = '/home/Documents/objectivescores11';

%% main logic
k=0;
for i=1:824
    disp(['Processing file : ', num2str(i)])
    
    % read clean and noisy file
    clean_file =[clean_path,file_clean(i).name];
    noisy_file = [noisy_path,file_noisy(i).name];
    [clean,fs] = audioread(clean_file);
    noisy = audioread(noisy_file);
    
    % get oracle mask
    [ impulse_response,Clean_gtm,Noisy_gtm,N_gtm,Y_filtered ] = my_gammatone( clean, noisy, fs, wlen, wshift, 0.95);
    IRM = (Clean_gtm.^2)./(Clean_gtm.^2 + N_gtm.^2);
     
    % load enhanced mask
    predictedMasks = load([pathpredicted_mask,'File_',num2str(i-1)]);
    Min_frame = min([length(Clean_gtm(1,:))]);
    multiple = ceil(Min_frame/patch_size);
    subtract_frames = abs(multiple*patch_size-Min_frame);
    Pred_mask = predictedMasks.squeezed_predmask;
    Pred_mask = Pred_mask(:,1:end-subtract_frames);
    
    % get oracle and enhanced waveforms
    oracle_recon = synthesis(W,wshift,IRM,Y_filtered,impulse_response, 0.95);
    pred_recon = synthesis(W,wshift,Pred_mask,Y_filtered,impulse_response, 0.95);
    
    % mean-var normalizing all the waveforms
    clean = (clean-mean(clean))/std(clean)/12;
    noisy = (noisy-mean(noisy))/std(noisy)/12;
    pred_recon = (pred_recon'-mean(pred_recon))/std(pred_recon)/12;
    oracle_recon = (oracle_recon'-mean(oracle_recon))/std(oracle_recon)/12;
    
    % store all the waveforms
    audiowrite([clean_wavform,'File_',num2str(i),'.wav'],clean(2:length(pred_recon)),fs)
    audiowrite([noisy_wavform,'File_',num2str(i),'.wav'],noisy(2:length(pred_recon)),fs)
    audiowrite([gan_wavform,'File_',num2str(i),'.wav'],pred_recon(1:end-1),fs)
    audiowrite([actual_wavform,'File_',num2str(i),'.wav'],oracle_recon(1:end-1),fs)
    
end

warning('off')
%% Initialization
number = 824;
OV=4;                         % overlap factor of 2 (4 is also often used)
wshift=80;                    % set frame increment in samples
wlen=wshift*OV;               % DFT window length
W=hamming(wlen);              % window
N_fft = 256;                  % n-point FFT
beta = 1;
fs = 16000;

%% load speech, gammatone energies, impulse response
% random = randperm(M);
for i=1:number
    display(['current file is : ',num2str(i)]);
    
    clean_file = [clean_wavform, 'File_', num2str(i),'.wav'];
    noisy_file = [noisy_wavform, 'File_', num2str(i),'.wav'];
    oracle_file = [path_orac_enhanced,'Oracle/File_',num2str(i),'.wav'];
    enhanced_file = [path_orac_enhanced,'Ganrmse/File_',num2str(i),'.wav'];
    
    % find stoi
    scores_stoi_oracle = stoi(clean_file, oracle_file,fs);
    scores_stoi_enhanced = stoi(clean_file, enhanced_file,fs);
    scores_stoi_noisy = stoi(clean_file,noisy_file,fs);
    
    % find composite scores
    [A1,A2,A3]=composite(clean_file,noisy_file);
    [B1,B2,B3]=composite(clean_file, enhanced_file);
    [C1,C2,C3]=composite(clean_file, oracle_file);
    
    % find pesq
    P1=pesq(clean_file,noisy_file);
    P2=pesq(clean_file, enhanced_file);
    P3=pesq(clean_file, oracle_file);
        
    % update statistics
    STOI_oracle(i) = scores_stoi_oracle;
    STOI_enhanced(i) = scores_stoi_enhanced;  
    STOI_noisy(i) = scores_stoi_noisy; 
    
    clc;
    
    PESQ_Noisy(i) = P1;
    PESQ_Enhanced(i)=P2;
    PESQ_Oracle(i)=P3; 
      
    SIG_Noisy(i) = A1;
    SIG_Enhanced(i) = B1;
    SIG_Oracle(i) = C1; 
    
    BAK_Noisy(i) = A2;
    BAK_Enhanced(i) = B2;
    BAK_Oracle(i) = C2; 
    
    MOS_Noisy(i) = A3;
    MOS_Enhanced(i) = B3;
    MOS_Oracle(i) = C3; 
    
    clc;
    
end

%% find mean of different measures
MEAN_SIG_oracle = mean(SIG_Oracle);
MEAN_SIG_enhanced = mean(SIG_Enhanced);
MEAN_SIG_noisy = mean(SIG_Noisy);

MEAN_BAK_oracle = mean(BAK_Oracle);
MEAN_BAK_enhanced = mean(BAK_Enhanced);
MEAN_BAK_noisy = mean(BAK_Noisy);

MEAN_OVL_oracle = mean(MOS_Oracle);
MEAN_OVL_enhanced = mean(MOS_Enhanced);
MEAN_OVL_noisy = mean(MOS_Noisy);

MEAN_PESQ_oracle = mean(PESQ_Oracle);
MEAN_PESQ_enhanced  = mean(PESQ_Enhanced);
MEAN_PESQ_noisy = mean(PESQ_Noisy);

MEAN_STOI_oracle = mean(STOI_oracle);
MEAN_STOI_enhanced  = mean(STOI_enhanced);
MEAN_STOI_noisy = mean(STOI_noisy);

%% save
Oracle.SIG = MEAN_SIG_oracle;
Oracle.BAK = MEAN_BAK_oracle;
Oracle.MOS = MEAN_OVL_oracle;
Oracle.pesq = MEAN_PESQ_oracle;
Oracle.stoi = MEAN_STOI_oracle;

Enhanced.SIG = MEAN_SIG_enhanced;
Enhanced.BAK = MEAN_BAK_enhanced;
Enhanced.MOS = MEAN_OVL_enhanced;
Enhanced.pesq = MEAN_PESQ_enhanced;
Enhanced.stoi = MEAN_STOI_enhanced;

Noisy.SIG = MEAN_SIG_noisy;
Noisy.BAK = MEAN_BAK_noisy;
Noisy.MOS = MEAN_OVL_noisy;
Noisy.pesq = MEAN_PESQ_noisy;
Noisy.stoi = MEAN_STOI_noisy;

save([wheretostore,'.mat'],'Noisy', 'Enhanced','Oracle');
