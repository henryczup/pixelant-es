clc
clear all
close all
tic

% net = importONNXNetwork('freq_scale_1_t_5_thick32_eff_60k.onnx');
net = importONNXNetwork('TLfwdmodel.onnx');
n = 1000;
m = 144; %4.5ghz
fmin = 1e9; fmax = 5e9; N = 81; freq = linspace(fmin,fmax,N);
center_fiu = find(freq == 3.6e9)  % Second band
% center_fil = find(freq == 4e9);  % First band
% % center_fiu = find(freq == 4.5e9); % Second band
% center_fiu = find(freq == 4.7e9);
num_pixels = 12;
%% Triple band
% pass_band = [center_fil-2:center_fil+2,center_fiu-2:center_fiu+2,center_fit-2:center_fit+2];
% stop_band = [1:center_fil-3,center_fil+3:center_fiu-3,center_fiu+3:center_fit-3,center_fit+3:81];

%% Single band
pass_band = [center_fiu-4:center_fiu+4];
stop_band = [1:center_fiu-5,center_fiu+5:81];

%% dual band
% pass_band = [center_fil-2:center_fil+2, center_fiu-2:center_fiu+2];
% stop_band = [1:center_fil-3,center_fil+3:center_fiu-3,center_fiu+3:81];


pass_freq = freq(pass_band);
stop_freq = freq(stop_band);

%% Projecting a vector onto a boolean vector using BPSO 
 
wmax=0.9; % inertia weight
wmin=0.4; % inertia weight
c1=2; % acceleration factor
c2=2; % acceleration factor
maxite=50;
maxrun=1;
tic
for run=1:maxrun
x = randi([0,1],n,m);
initial_inputmatrix = x; % initial population
v = 0.1*initial_inputmatrix; % initial velocity
for i=1:n
X_t = reshape(initial_inputmatrix(i,:),12,12);
X_t(6:7,1) = 1;
Ypr1 = predict(net,X_t);
% esterr = norm(Ypr1 - ss',2)/norm(ss',2);
 for jj = 1:length(freq)
      if(abs(Ypr1(1,jj)) > 10)
        calc_r(i,jj) = 10;
      elseif(abs(Ypr1(1,jj)) < 5)
          calc_r(i,jj) = 0;
      else
          calc_r(i,jj) = abs(Ypr1(1,jj));
      end
 end
Error_vec(i,1) = calculatefit(Ypr1(1,:),pass_freq,stop_freq,calc_r(i,:),freq); 

end
[fmin0,index0]=max(Error_vec);
pbest=initial_inputmatrix; % initial pbest
gbest=initial_inputmatrix(index0,:); % initial gbest

ite=1;
while ite<=maxite
%
w=wmax-(wmax-wmin)*ite/maxite; 
for i=1:n
	for j=1:m
        if (gbest(j) == pbest(i,j) && pbest(i,j) == 1)
            v(i,j)=w*v(i,j) + c1*rand() + c2*rand();
        elseif (gbest(j) == pbest(i,j) && pbest(i,j) == 0)
            v(i,j)=w*v(i,j) - c1*rand() - c2*rand();
            else
            v(i,j)=w*v(i,j);   %%% - c1*rand() - c2*rand();
        end
	end
end
%
%
%% Binary PSO
for i=1:n
    for j=1:m
        v_prob(i,j)= 1/(1 + exp(-v(i,j)));
    end
end

x_probs = rand(n,m);
for i=1:n
    for j=1:m
        if x_probs(i,j) < v_prob(i,j)
            xx(i,j) = 1;
        else 
            xx(i,j) = 0;
        end
    end
end

%% evaluating fitness
for i=1:n
tt = reshape(xx(i,:),12,12);
tt(6:7,1) = 1;
 Ypr2 = predict(net,tt);
 for jj = 1:length(freq)
      if(abs(Ypr2(1,jj)) > 10)
        calc_r(i,jj) = 10;
      elseif(abs(Ypr2(1,jj)) < 5)
          calc_r(i,jj) = 0;
      else
          calc_r(i,jj) = abs(Ypr2(1,jj));
      end
 end
f(i,1) = calculatefit(Ypr2(1,:),pass_freq,stop_freq,calc_r(i,:),freq); 

end
%% updating pbest and fitness
for i=1:n
if f(i,1)>Error_vec(i,1)
pbest(i,:)=xx(i,:);
Error_vec(i,1)=f(i,1);
end
end
[fmin,index]=max(Error_vec);
ffmin(ite,run)=fmin;
ffite(run)=ite;
if fmin>fmin0
gbest=pbest(index,:);
fmin0=fmin;
end
if ite==1
disp(sprintf('Iteration Best particle Objective fun'));
end
disp(sprintf('%8g %8g %8.4f',ite,index,fmin0));
ite=ite+1;
end
%% pso algorithm-----------------------------------------------------end
gbest
antenna_de = reshape(gbest,12,12);
output_ne = predict(net,antenna_de);
for jj = 1:length(freq)
      if(abs(output_ne(1,jj)) > 10)
        calc_r(1,jj) = 10;
      elseif(abs(output_ne(1,jj)) < 5)
          calc_r(1,jj) = 0;
      else
          calc_r(1,jj) = abs(output_ne(1,jj));
      end
 end
err = calculatefit(output_ne(1,:),pass_freq,stop_freq,calc_r(1,:),freq); 
% err = norm(output_ne - ss',2)/norm(ss',2);
fff(run)=err;
rgbest(run,:)=gbest;
disp(sprintf('--------------------------------------'));
end
toc
disp(sprintf('\n'));
disp(sprintf('*********************************************************'));
disp(sprintf('Final Results-----------------------------'));
[bestfun,bestrun]=min(fff)
best_variables=rgbest(bestrun,:)
disp(sprintf('*********************************************************'));
% toc
% BPSO convergence characteristic
antenna_des = reshape(best_variables,12,12)
output_new = predict(net,antenna_des);
% err = norm(output_new - ss',2)/norm(ss',2);
plot(ffmin(1:ffite(bestrun),bestrun),'-k');
xlabel('Iteration');
ylabel('Fitness function value');
title('PSO convergence characteristic')
h = figure
figure
plot(freq,output_new(1,1:81))
% plot(freq,output_new(1,:),freq,ss')
legend("Reconstructed","Original","FontSize",14,"Location","northeast");
xlabel("freq","FontSize",16)
ylabel("Return Loss","FontSize",16)
print(h,'fig.png','-dpng')
des_ant = antenna_des
% save('antenna_tl_1_t_5_fr4_35.mat','des_ant','ffmin','output_new','fff');
toc
function fit_value = calculatefit(s_params,p_freq,s_freq,calc_r,freq)
        tot_p = 0; tot_s = 0;
        for ii = 1:length(p_freq)
                freq_index = find(freq == p_freq(ii));
                tot_p = tot_p + calc_r(freq_index);
   
        end
        
        for ii = 1:length(s_freq)
                freq_index = find(freq == s_freq(ii));
                tot_s = tot_s + (10 - calc_r(freq_index));
   
        end
        fit_value = tot_p + tot_s;
end