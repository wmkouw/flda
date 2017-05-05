function run_daexp_digits(clf,cc)
% Stupid script to run all combinations
disp(['Running crossvalidated classification error experiments on the digits dataset']);

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
nE = 50;
Kt = 'rbf';
Kp = [1 10];
mu = [0.1 1];
La = 0;
Ga = [1e-3 1e-1];
l2 = [1e-2 1 100 1000];
set = 'mm_rot_small';

daexp_digits(clf, 'prep', prep, 'cix', cc, 'nR', nR, 'nF', nF, 'l2', l2, 'nE', nE, 'mu', mu,'La', La, 'Ga', Ga, 'Kt',Kt,'Kp',Kp,'set',set);

% exit;

end
