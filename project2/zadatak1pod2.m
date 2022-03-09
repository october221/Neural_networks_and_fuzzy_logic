clc, clear, close all
rng(200);

podaci = readtable('Rice.csv');
ulaz = [podaci.AREA, podaci.PERIMETER, podaci.MAJORAXIS, podaci.MINORAXIS, podaci.ECCENTRICITY, podaci.CONVEX_AREA, podaci.EXTENT];
izlaz = podaci.CLASS;

ulaz = ulaz';
izlaz = izlaz';

izlazBrojevi = [];

for i = izlaz
    k = 0;
    if strcmp(i,'Cammeo')
        k = 1;
    else
    if strcmp(i,'Osmancik')
        k = 2;
    else
    if strcmp(i,'Kecimen')
        k = 3;
    end
    end
    end
    izlazBrojevi = [izlazBrojevi, k];
end

figure, histogram(izlazBrojevi)

%% podela ulaznih podataka na trening, test i validaciju
rng(200);

izlazOneHot = zeros(3, length(izlazBrojevi));
izlazOneHot(1, izlazBrojevi == 1) = 1;
izlazOneHot(2, izlazBrojevi == 2) = 1;
izlazOneHot(3, izlazBrojevi == 3) = 1;

K1 = ulaz(:, izlazBrojevi == 1);
K2 = ulaz(:, izlazBrojevi == 2);
K3 = ulaz(:, izlazBrojevi == 3);
izlaz1 = izlazOneHot(:, izlazBrojevi == 1);
izlaz2 = izlazOneHot(:, izlazBrojevi == 2);
izlaz3 = izlazOneHot(:, izlazBrojevi == 3);

N1 = length(K1);
N2 = length(K2);
N3 = length(K3);

K1trening = K1(:, 1 : 0.7*N1);
K1test = K1(:, 0.7*N1 + 1 : 0.85*N1);
K1val = K1(:, 0.85*N1 + 1 : N1);
izlaz1trening = izlaz1(:, 1 : 0.7*N1);
izlaz1test = izlaz1(:, 0.7*N1 + 1 : 0.85*N1);
izlaz1val = izlaz1(:, 0.85*N1 + 1 : N1);

K2trening = K2(:, 1 : 0.7*N2);
K2test = K2(:, 0.7*N2 + 1 : 0.85*N2);
K2val = K2(:, 0.85*N2 + 1 : N2);
izlaz2trening = izlaz2(:, 1 : 0.7*N2);
izlaz2test = izlaz2(:, 0.7*N2 + 1 : 0.85*N2);
izlaz2val = izlaz2(:, 0.85*N2 + 1 : N2);

K3trening = K3(:, 1 : 0.7*N3);
K3test = K3(:, 0.7*N3 + 1 : 0.85*N3);
K3val = K3(:, 0.85*N3 + 1 : N3);
izlaz3trening = izlaz3(:, 1 : 0.7*N3);
izlaz3test = izlaz3(:, 0.7*N3 + 1 : 0.85*N3);
izlaz3val = izlaz3(:, 0.85*N3 + 1 : N3);

ulazTrening = [K1trening, K2trening, K3trening];
izlazTrening = [izlaz1trening, izlaz2trening, izlaz3trening];

ind = randperm(length(izlazTrening));
ulazTrening = ulazTrening(:, ind);
izlazTrening = izlazTrening(:, ind);

ulazTest = [K1test, K2test, K3test];
izlazTest = [izlaz1test, izlaz2test, izlaz3test];

ulazVal = [K1val, K2val, K3val];
izlazVal = [izlaz1val, izlaz2val, izlaz3val];

ulazSve = [ulazTrening, ulazVal];
izlazSve = [izlazTrening, izlazVal];

%%
rng(200);
arhitektura = {[5,5], [4,4,4], [6,3,3], [7, 4], [6,6,2], [3,3,4], [5,4,4]};
weights = {[2,1,5], [3,2,10], [1,2,3], [2,5,1], [4,3,16]};
regularizacija = {0.1, 0.3, 0.5, 0.7, 0.9};
najbolja_arh = 0;
najbolji_F1 = 0;
najbolja_reg = 0;
najbolja_tezina = [];
najbolja_epoha = 0;

for arh = 1 : length(arhitektura)
    for reg = 1 : length(regularizacija)
        for w = 1 : length(weights)
        
           net = patternnet(arhitektura{arh});
           net.divideFcn = 'divideind';
           net.divideParam.trainInd = 1 : length(ulazTrening);
           net.divideParam.valInd = length(ulazTrening) + 1 : length(ulazSve);
           net.divideParam.testInd = [];
           
           for i = 1 : length(arhitektura{arh})
                net.layers{i}.transferFcn = 'tansig';
           end
           net.layers{i+1}.transferFcn = 'softmax';

           net.trainFcn = 'trainscg';
           net.performParam.regularization = regularizacija{reg};

           net.trainParam.epochs = 1000;
           net.trainParam.goal = 1e-4;
           net.trainParam.min_grad = 1e-5;
           
           wgh = weights{w};

           weight = ones(3, length(izlazSve));
           wg1 = wgh(1);
           wg2 = wgh(2);
           wg3 = wgh(3);
           arr = izlazSve == [1,0,0]';
           arr = arr(1,:);
           weight(:,arr) = wg1;
           arr = izlazSve == [0,1,0]';
           arr = arr(2,:);
           weight(:,arr) = wg2;
           arr = izlazSve == [0,0,1]';
           arr = arr(3,:);
           weight(:,arr) = wg3;

           [net, info] = train(net, ulazSve, izlazSve, [], [], weight);

           pred = sim(net, ulazVal);
           [e, cm] = confusion(izlazVal, pred);
           
           TP1 = cm(1,1);
           TN1 = cm(2,2) + cm(3,3) + cm(2,3) + cm(3,2);
           FP1 = cm(1,2) + cm(1,3);
           FN1 = cm(2,1) + cm(3,1);
           precision1 = TP1 / (TP1 + FP1);
           recall1 = TP1 / (TP1 + FN1);
           F11 = 2 * (precision1 * recall1 / (precision1 + recall1)); 
           
           TP2 = cm(2,2);
           TN2 = cm(1,1) + cm(1,3) + cm(3,1) + cm(3,3);
           FP2 = cm(2,1) + cm(2,3);
           FN2 = cm(1,2) + cm(3,2);
           precision2 = TP2 / (TP2 + FP2);
           recall2 = TP2 / (TP2 + FN2);
           F12 = 2 * (precision2 * recall2 / (precision2 + recall2));
           
           TP3 = cm(3,3);
           TN3 = cm(1,1) + cm(1,2) + cm(2,1) + cm(2,2);
           FP3 = cm(3,1) + cm(3,2);
           FN3 = cm(1,3) + cm(2,3);
           precision3 = TP3 / (TP3 + FP3);
           recall3 = TP3 / (TP3 + FN3);
           F13 = 2 * (precision3 * recall3 / (precision3 + recall3));
           
           F1 = (F11 + F12 + F13) / 3;

           if F1 > najbolji_F1
               najbolji_F1 = F1;
               najbolja_arh = arhitektura{arh};
               najbolja_reg = reg;
               najbolja_tezina = weight;
               najbolja_epoha = info.best_epoch;
           end
           
        end
    end
end

%%
disp("Najbolja arhitektura je: ");
disp(najbolja_arh);
disp("Najbolja regularizacija je: " + regularizacija{najbolja_reg});
wu = unique(najbolja_tezina);
disp("Najbolja tezina je: ");
disp(wu);
%% obucavanje najbolje mreze
rng(200);

net = patternnet(najbolja_arh);
net.divideFcn = '';
for i = 1 : length(najbolja_arh)
     net.layers{i}.transferFcn = 'tansig';
end
net.layers{i+1}.transferFcn = 'softmax';
net.performParam.regularization = regularizacija{najbolja_reg};
net.trainFcn = 'trainscg';

net.trainParam.epochs = najbolja_epoha;
net.trainParam.goal = 1e-4;
net.trainParam.min_grad = 1e-5;

[net, info] = train(net, ulazSve, izlazSve, [], [], najbolja_tezina);

pred = sim(net, ulazTest);
[e, cm] = confusion(izlazTest, pred);
figure, plotconfusion(izlazTest, pred), title("Matrica konfuzije za test skup");

TP1 = cm(1,1);
TN1 = cm(2,2) + cm(3,3) + cm(2,3) + cm(3,2);
FP1 = cm(1,2) + cm(1,3);
FN1 = cm(2,1) + cm(3,1);
precision1 = TP1 / (TP1 + FP1);
recall1 = TP1 / (TP1 + FN1);
F11 = 2 * (precision1 * recall1 / (precision1 + recall1)); 

TP2 = cm(2,2);
TN2 = cm(1,1) + cm(1,3) + cm(3,1) + cm(3,3);
FP2 = cm(2,1) + cm(2,3);
FN2 = cm(1,2) + cm(3,2);
precision2 = TP2 / (TP2 + FP2);
recall2 = TP2 / (TP2 + FN2);
F12 = 2 * (precision2 * recall2 / (precision2 + recall2));

TP3 = cm(3,3);
TN3 = cm(1,1) + cm(1,2) + cm(2,1) + cm(2,2);
FP3 = cm(3,1) + cm(3,2);
FN3 = cm(1,3) + cm(2,3);
precision3 = TP3 / (TP3 + FP3);
recall3 = TP3 / (TP3 + FN3);
F13 = 2 * (precision3 * recall3 / (precision3 + recall3));

F1 = (F11 + F12 + F13) / 3;

disp("Preciznost za prvu klasu je: " + precision1);
disp("Osetljivost za prvu klasu je: " + recall1);
disp("Preciznost za drugu klasu je: " + precision2);
disp("Osetljivost za drugu klasu je: " + recall2);
disp("Preciznost za trecu klasu je: " + precision3);
disp("Osetljivost za trecu klasu je: " + recall3);

predtr = sim(net, ulazSve);
figure, plotconfusion(izlazSve, predtr), title("Matrica konfuzije za trening skup");

figure, plotperform(info);



