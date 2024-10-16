clearvars
close all
tic
%https://www.analyticsvidhya.com/blog/2021/07/image-denoising-using-autoencoders-a-beginners-guide-to-deep-learning-project/

% gera (NumeroTri) ondas triangulares com diferentes delays
% frequencia/duty cycle dado por pzeros e pones

tamanho=1000;
tamanhoMaior = 2000;
NumeroTri=2000;
%onda quadrada
tempo=0:1:tamanho-1;
psubida = 20;
pdescida  = 20;
AmpRuido = 1;

TriCell = horzcat((1/psubida).*tempo(1:psubida), ...
1-(1/pdescida).*tempo(1:pdescida));

TriGrandao = repmat(TriCell,...
[1,round(tamanhoMaior/length(TriCell)),1]);

figure(1)
plot(TriGrandao,'-x')

for count=1:NumeroTri
  p= round(tamanho*0.5*rand(1));
  triangulo(count,:) = TriGrandao(p+1:tamanho+p-1+1);
end




%csvwrite('triangulo.csv', triangulo)
csvwrite('Lixo.csv', triangulo)

% adiciona ruido
for count=1:NumeroTri
  p= round(tamanho*0.5*rand(1));
  TriNoise(count,:) = -0.5 + AmpRuido*rand(1,tamanho) + triangulo(count,:);
end



%csvwrite('TriNoise.csv', TriNoise)
csvwrite('Lixo.csv', TriNoise)



figure(2)
plot((1:1:1000), triangulo(11,:), 'r', (1:1:1000), TriNoise(11,:) ,'k', 'Linewidth', 1)
ylabel('Amplitude','FontSize',12,'Interpreter','Latex')
texto={'Clean', 'Noisy'};
legend(texto,'Interpreter','latex','FontSize',12, 'Location', 'best');
%xlabel('Frequency [MHz]','FontSize',12,'Interpreter','Latex')
%ylabel('Electric Field inside the cavity [dBm]','Interpreter','latex','FontSize',12)
set(gca,'Fontname','Latin Modern Roman','FontSize',12)

return



% calculo SNR

SNR= 10.*log10(sum(triangulo.^2,2)./(sum(TriNoise.^2,2)));
figure(2)
plot(SNR, 'Linewidth',2)
ylabel('SNR [dB]')
xlabel('Waveform sample')


figure(3)
histogram(SNR,'Facecolor', 'r')
xlabel('SNR [dB]','Interpreter','latex','FontSize',12)
ylabel('# samples','FontSize',12,'Interpreter','Latex')
%texto={'gap=0.1 mm', 'gap=0.55 mm', 'gap=1 mm'};
%legend(texto,'Interpreter','latex','FontSize',12, 'Location', 'best');
%xlabel('Frequency [MHz]','FontSize',12,'Interpreter','Latex')
%ylabel('Electric Field inside the cavity [dBm]','Interpreter','latex','FontSize',12)
set(gca,'Fontname','Latin Modern Roman','FontSize',12)



toc
