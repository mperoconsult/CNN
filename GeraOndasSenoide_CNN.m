clearvars
close all
%https://www.analyticsvidhya.com/blog/2021/07/image-denoising-using-autoencoders-a-beginners-guide-to-deep-learning-project/

% gera (NumeroSenoide) ondas senoidais com diferentes delays
% frequecia varia 
tic


tamanho=1000;
tamanhoMaior = 2000;
NumeroSenoide= 2000;
fmin = 0.1;
fmax= 10;
delayMax = 2*pi();
AmpRuido = 1;

tempo=linspace(0,2*pi,tamanho);
atraso = delayMax*rand(1,NumeroSenoide)';

freq=(fmin + fmax*rand(1,NumeroSenoide))';
Seno = real(exp(1i*2*pi.*freq.*tempo+1i.*atraso));

figure(1)
plot(tempo, Seno(1,:),'-x', tempo, Seno(2,:),'g', tempo, Seno(3,:),'k',tempo, Seno(4,:),'--k')

%csvwrite('Seno.csv', Seno)
csvwrite('Lixo.csv', Seno)



% adiciona ruido
for count=1:NumeroSenoide
  p= round(tamanho*0.5*rand(1));
  SenoNoise(count,:) = -0.5+ AmpRuido*rand(1,tamanho) + Seno(count,:);
end


figure(2)
% plot(tempo, SenoNoise(11,:), '-xg',...
% tempo, Seno(11,:), '-g',...
% tempo, SenoNoise(13,:), '-xk', ...
% tempo, Seno(13,:), '-k', 'Linewidth', 2)
% xlabel('time','Interpreter','latex','FontSize',12)
plot(tempo, Seno(11,:), 'r', tempo, SenoNoise(11,:),'k', 'Linewidth', 1)
ylabel('Amplitude','FontSize',12,'Interpreter','Latex')
texto={'Clean', 'Noisy'};
legend(texto,'Interpreter','latex','FontSize',12, 'Location', 'best');
%xlabel('Frequency [MHz]','FontSize',12,'Interpreter','Latex')
%ylabel('Electric Field inside the cavity [dBm]','Interpreter','latex','FontSize',12)
set(gca,'Fontname','Latin Modern Roman','FontSize',12)



return

%csvwrite('SenoNoise.csv', SenoNoise)
csvwrite('Lixo.csv', SenoNoise)


figure(4)
plot(tempo, SenoNoise(110,:), '-rx', tempo, Seno(110,:),'-k')

% calculo SNR

SNR= 10.*log10(sum(Seno.^2,2)./(sum(SenoNoise.^2,2)));
figure(3)
plot(SNR, 'Linewidth',2)
ylabel('SNR [dB]')
xlabel('Waveform sample')


figure(3)
histogram(SNR,'Facecolor', 'g')
xlabel('SNR [dB]','Interpreter','latex','FontSize',12)
ylabel('# samples','FontSize',12,'Interpreter','Latex')
%texto={'gap=0.1 mm', 'gap=0.55 mm', 'gap=1 mm'};
%legend(texto,'Interpreter','latex','FontSize',12, 'Location', 'best');
%xlabel('Frequency [MHz]','FontSize',12,'Interpreter','Latex')
%ylabel('Electric Field inside the cavity [dBm]','Interpreter','latex','FontSize',12)
set(gca,'Fontname','Latin Modern Roman','FontSize',12)
toc