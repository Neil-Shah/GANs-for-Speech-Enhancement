%% Code Information
% This MATLAB code is used to compute the context feture from the basic
% static feature vector,

%% Data definition
% cep := basic static feature
% ND := Dimension of feature vector
% NLcon := Number of context from left side
% NRcon := Number of context from right side

function cep1=framecontext(cep,ND,NLcon,NRcon)

if nargin<1; error('Oh Boy ! Give correct number of inputs..'); end
if nargin<2; if(size(cep,1)<size(cep,2));      ND=size(cep,1);    else ND=size(cep,2);     end;end
if nargin<3;NLcon=2; end
if nargin<4;NRcon=2; end

NF=size(cep,2);         % Number of frames
cep1=zeros(ND*(NLcon+NRcon),NF); % Preallocate feature size

%% Feature dimension
if size(cep,1)~=ND % If num rows are not the dimension of feature vector
    cep=cep';
end

%% Do frame repeatation part
xt=zeros(ND,NF+(NLcon+NRcon)); % Preallocate feature size appended
for k=1:NLcon                       % Store starting frames
    xt(:,k)=cep(:,1);
end

for k=NLcon+1:NLcon+NF           % Store core frames
    xt(:,k)=cep(:,k-NLcon);
end

for k=1+NLcon+NF:NF+(NLcon+NRcon)  % Store Ending frames
    xt(:,k)=cep(:,end);
end

x1=xt;                          % Assign to new variable

%% Do combine
for k=1:NF              % For every frame
    for k1=1:(NLcon+NRcon+1)
        cep1((k1-1)*ND+1:k1*ND,k)=x1(:,k+k1-1);
    end
end