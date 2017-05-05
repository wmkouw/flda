function run_daexp_office(clf,cc)
% Stupid script to run all combinations
disp(['Running crossvalidated classification error experiments on the office dataset']);

addpath(genpath('../minFunc'));
addpath(genpath('../tcaPackage'));
addpath(genpath('../DA_SA'));
addpath(genpath('../libsvm-3.21'));
addpath(genpath('../da-tools'));

addpath(genpath('..\minFunc'));
addpath(genpath('..\..\da-tools'));
addpath(genpath('..\..\da-tools\tcaPackage'));
addpath(genpath('..\..\da-tools\DA_SA'));
addpath(genpath('C:\Users\Wouter\Dropbox\Codes\libsvm-3.21\matlab'));

fts = 'caltech';
prep = {'max'};
nR = 2;
nF = 2;
nE = 500;
Kt = 'rbf';
Kp = [1 10 100];
mu = [0.1 1];
La = [0];
Ga = [.001 .01 .1 1];
l2 = [0 1e-6 1e-2 1e-1 1 1e1 1e2 1e3 1e4 1e5 1e6];

daexp_office(clf, 'prep', prep, 'cix', cc, 'nR', nR, 'nF', nF, 'l2', l2, 'fts', fts, 'nE', nE, 'mu', mu, 'La',La,'Ga',Ga,'Kt',Kt,'Kp',Kp);

% exit

end


%%
