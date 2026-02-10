function generateantenna_for_tandem_air(x_dis, y_dis, samples_to_generate)

    parpool('local',8)
    iter = samples_to_generate;
    tic
    e_eff = 1; %% for air
    L = (2*3.75e-3)/sqrt(e_eff); %% x axis //Conventional size for resonating at %5 GHz
    W = (2*3.75e-3)/sqrt(e_eff); %% y axis
    nx = x_dis; %%cells along x = nx-1,Number of columns of Exc matrix
    ny = y_dis; %%cells along y = ny-1 %Number of rows in Exc Matrix
    gndLength = 3*L;
    gndWidth = 3*W; %%pcbTraceWidth = 0.662e-3; % for 0.305 ro4003
    pcbTraceWidth = 2*0.988e-3;
    pcbTraceLength = 0.5*(gndLength - L);
    x = linspace(-L/2,L/2,nx); %bottom left coordinates of each cell
    y = linspace(-W/2,W/2,ny); %bottom left coordinates of each cell
    dx = x(2) - x(1); px = 1.25*dx; % 5% bigger in size(for fabrication purpose)
    dy = y(2) - y(1); py = 1.25*dy;
    first_one = true;
    num_pixel = (ny-1)*(nx-1);
    Test_patches = zeros(iter,num_pixel);
    spec = zeros(iter,81);

    tmp = zeros(iter,num_pixel); tmp_spec = zeros(1,81);
    fmin = 10e9; fmax = 20e9; N = 81; freq = linspace(fmin,fmax,N);   
    lambda = 3e8/freq(end);
    output_file_name = ['antenna_dataset_air_', num2str(num_pixel), '_p_hf_',...
        num2str(samples_to_generate), '.mat'];
    disp(output_file_name);

    parfor ii=1:samples_to_generate
         ii
         tic 
         % Exc = reshape(ant_des(ii,:,:,:),[12,12]);
         Exc = randi([0,1],[12,12]);
         Exc1 = reshape(Exc,1,[]);
         tmp(ii,:) = Exc1; 
        p = generateantenna_air(nx, ny, Exc);
        figure
        try
        mesh(p,'MaxEdgeLength',lambda/15)
        figure
        show(p)
        s = sparameters(p,freq,50);
        s11Fig = figure;
        rfplot(s,1,1);
        S11=rfparam(s,1,1);
        s11_mag = abs(S11);
        s11_db = 20*log10(s11_mag);
        tmp_spec(ii,:)=s11_db';
        spec = tmp_spec(ii,:);
        catch
            disp('invalid');
        toc
        end
    Test_patches = tmp(ii,:);
    parsave(sprintf('output%d.mat', ii),Test_patches,spec);  
    end
end