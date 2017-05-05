function run_daexp_amazon(clf,cc)
% Stupid script to run all combinations
disp(['Running crossvalidated classification error experiments on the amazon dataset']);

addpath(genpath('../tcaPackage'));
addpath(genpath('../DA_SA'));
addpath(genpath('../libsvm-3.21'));
addpath(genpath('../da-tools'));

addpath(genpath('..\..\da-tools'));
addpath(genpath('..\..\da-tools\tcaPackage'));
addpath(genpath('..\..\da-tools\DA_SA'));
addpath(genpath('C:\Users\Wouter\Dropbox\Codes\libsvm-3.21\matlab'));

prep = {'max'};
nR = 1;
nF = 2;
nE = 500;
Kt = 'rbf';
Kp = [.001 1 1000];
mu = [0.01 0.1 1 10];
La = [0 1 10];
Ga = [.00001 0.001 1];
l2 = [1 10 100 1000 10000 100000];

daexp_amazon(clf, 'prep', prep, 'cix', cc, 'nR', nR, 'nF', nF, 'l2', l2, 'nE', nE, 'mu', mu,'Kt',Kt,'Kp',Kp,'Ga',Ga,'La',La);

% exit;

end
