function run_daexp_imdb(clf)
disp(['Running crossvalidated classification error experiments on the imdb dataset']);

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

prep = {'tf-idf','norm_samp','max'};
nR = 1;
nF = 2;
nE = 500;
Kt = 'rbf';
Kp = [1 10 100];
mu = [0.1 1];
La = [0 1];
Ga = [1e-4 1e-3 1e-2 1e-1 1];
L2 = [0 1e-3 1 1e1 1e2 1e3 1e4];

daexp_imdb(clf, prep, 'nR', nR, 'nF', nF, 'l2', L2, 'nE', nE, 'mu', mu,'Kt',Kt,'Kp',Kp,'Ga',Ga,'La',La);

% exit;

end
