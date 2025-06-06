% IF_Mstar.m
% Usage: drat = IF_Mstar(Mach <, gamma>) where gamma=1.4 if omitted
% Function for characteristic Mach number
% S. Collicott, Sep. 2009
% Copyright (C) 2010, Purdue University
%
function [Mstar] = IF_Mstar(arg1, varargin)
gam = 1.4;  % default ratio of specific heats
Mstar = NaN;
Mach = arg1; % Mach number
if isempty(find(Mach < 0)) && length(varargin) < 2
  if length(varargin)==1
    tmp = cell2mat(varargin);
    if tmp > 0
      gam = tmp;
    else
      gam = NaN;
    end
  end
  %
  Mstar = sqrt((gam+1)*Mach.^2 ./ (2+(gam-1)*Mach.^2));
end
