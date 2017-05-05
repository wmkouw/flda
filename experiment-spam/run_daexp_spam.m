function run_daexp_spam(clf,cc)
% Stupid script to run all combinations
disp(['Running crossvalidated classification error experiments on the spam dataset']);

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
Kp = 10;
La = 1;
Ga = 1;
mu = 0.1;
l2 = [1 10 100 1000 10000];

daexp_spam(clf, 'prep', prep, 'cix', cc, 'nR', nR, 'nF', nF, 'l2', l2, 'nE', nE, 'La', La, 'Ga', Ga, 'mu', mu,'Kt',Kt,'Kp',Kp);

% exit;

end
