warning('off', 'Octave:shadowed-function');
addpath("./var_bvar","./util","./distrib")

function out = str2mat(varargin)
  % Wrapper for legacy LeSage code calling deprecated str2mat
  out = char(varargin{:});
end

vnames = strvcat('il','in','ky','mi','oh','pa','tn','wv');
y = load('data/test.dat'); % use all eight states
nlag = 2;
tight = 0.1; % hyperparameter values
weight = 0.5;
decay = 1.0;
result = bvar(y,nlag,tight,weight,decay);
prt(result,vnames);
