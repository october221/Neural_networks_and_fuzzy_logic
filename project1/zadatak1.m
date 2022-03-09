clc, clear, close all
rng(200)

matrix = load('dataset1.mat');
matrix = cell2mat(struct2cell(matrix));

ulaz = matrix(:, 1:2);
izlaz = matrix(:, 3);

ulaz = ulaz';
izlaz = izlaz';

%% podela ulaza na klase

K1 = ulaz(:, izlaz == 1);
K2 = ulaz(:, izlaz == 2);
K3 = ulaz(:, izlaz == 3);

%% one hot
onehot = zeros(3, length(izlaz));
onehot(1, izlaz == 1) = 1;
onehot(2, izlaz == 2) = 1;
onehot(3, izlaz == 3) = 1;

figure, hold all
plot(K1(1, :), K1(2, :), 'b*')
plot(K2(1, :), K2(2, :), 'rd')
plot(K3(1, :), K3(2, :), 'yo')
legend({'Klasa 1','Klasa 2','Klasa 3'},'Location','southwest')
%% podela na trening i test skupove
N = length(izlaz);
ind = randperm(N);
indTrening = ind(1 : 0.8*N);
indTest = ind(0.8*N + 1 : N);

ulazTrening = ulaz(:, indTrening);
izlazTrening = onehot(:, indTrening);

ulazTest = ulaz(:, indTest);
izlazTest = onehot(:, indTest);

%% prva neuralna mreza - underfitting
slojevi1 = [2, 1];
net1 = patternnet(slojevi1);
net1.divideFcn = '';

for i = 1 : length(slojevi1)
	net1.layers{i}.transferFcn = 'tansig';
end
net1.layers{i+1}.transferFcn = 'softmax';

net1.trainParam.epochs = 1500;
net1.trainParam.goal = 1e-4;
net1.trainParam.min_grad = 1e-5;

%% druga neuralna mreza - optimalna
slojevi2 = [4, 3, 3];
net2 = patternnet(slojevi2);
net2.divideFcn = '';

for i = 1 : length(slojevi2)
	net2.layers{i}.transferFcn = 'tansig';
end
net2.layers{i+1}.transferFcn = 'softmax';

net2.trainParam.epochs = 1500;
net2.trainParam.goal = 1e-4;
net2.trainParam.min_grad = 1e-5;

%% treca neuralna mreza - overfitting
slojevi3 = [30, 20, 20];
net3 = patternnet(slojevi3);
net3.divideFcn = '';

for i = 1 : length(slojevi3)
	net3.layers{i}.transferFcn = 'tansig';
end
net3.layers{i+1}.transferFcn = 'softmax';

net3.trainParam.epochs = 1500;
net3.trainParam.goal = 1e-4;
net3.trainParam.min_grad = 1e-5;

%% treniranje za underfitting
net1 = train(net1, ulazTrening, izlazTrening);

%% treniranje za optimalnu
net2 = train(net2, ulazTrening, izlazTrening);

%% treniranje za overfitting
net3 = train(net3, ulazTrening, izlazTrening);

%% performanse za underfitting
pred1 = sim(net1, ulazTest);
figure, plotconfusion(izlazTest, pred1), title("Underfitting test skup");
predtr1 = sim(net1, ulazTrening)
figure, plotconfusion(izlazTrening, predtr1), title("Underfitting trening skup");

%% performanse za optimalan
pred2 = sim(net2, ulazTest);
figure, plotconfusion(izlazTest, pred2), title("Optimalna mreza test skup");
predtr2 = sim(net2, ulazTrening)
figure, plotconfusion(izlazTrening, predtr2), title("Optimalna mreza trening skup");

%% performanse za overfitting
pred3 = sim(net3, ulazTest);
figure, plotconfusion(izlazTest, pred3), title("Overfitting test skup");
predtr3 = sim(net3, ulazTrening)
figure, plotconfusion(izlazTrening, predtr3), title("Overfitting trening skup");

%% granice odlucivanja
Ntest = 1000;
ulazgr = [];
x1 = linspace(-5,5, Ntest);
x2 = linspace(-5,5, Ntest);

for x11 = x1
	pom = [x11*ones(1, Ntest); x2];
	ulazgr = [ulazgr, pom];
end

predgr1 = sim(net1, ulazgr);
[vr1, klasa1] = max(predgr1);

predgr2 = sim(net2, ulazgr);
[vr2, klasa2] = max(predgr2);

predgr3 = sim(net3, ulazgr);
[vr3, klasa3] = max(predgr3);

% underfitting

K1g1 = ulazgr(:, klasa1 == 1);
K2g1 = ulazgr(:, klasa1 == 2);
K3g1 = ulazgr(:, klasa1 == 3);

figure, hold all
plot(K1g1(1,:), K1g1(2,:), '.')
plot(K2g1(1,:), K2g1(2,:), '.')
plot(K3g1(1,:), K3g1(2,:), '.')
plot(K1(1, :), K1(2, :), 'b*')
plot(K2(1, :), K2(2, :), 'rd')
plot(K3(1, :), K3(2, :), 'yo')


% optimalna

K1g2 = ulazgr(:, klasa2 == 1);
K2g2 = ulazgr(:, klasa2 == 2);
K3g2 = ulazgr(:, klasa2 == 3);

figure, hold all
plot(K1g2(1,:), K1g2(2,:), '.')
plot(K2g2(1,:), K2g2(2,:), '.')
plot(K3g2(1,:), K3g2(2,:), '.')
plot(K1(1, :), K1(2, :), 'b*')
plot(K2(1, :), K2(2, :), 'rd')
plot(K3(1, :), K3(2, :), 'yo')

% overfitting

K1g3 = ulazgr(:, klasa3 == 1);
K2g3 = ulazgr(:, klasa3 == 2);
K3g3 = ulazgr(:, klasa3 == 3);

figure, hold all
plot(K1g3(1,:), K1g3(2,:), '.')
plot(K2g3(1,:), K2g3(2,:), '.')
plot(K3g3(1,:), K3g3(2,:), '.')
plot(K1(1, :), K1(2, :), 'b*')
plot(K2(1, :), K2(2, :), 'rd')
plot(K3(1, :), K3(2, :), 'yo')








