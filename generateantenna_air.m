function p = generateantenna_air(x_dis, y_dis, ant_des)
            tic
            e_eff = 1; %for air
            L = ((2*3.75e-3)/sqrt(e_eff)) % x axis //Conventional size for resonating at %5 GHz
            W = ((2*3.75e-3)/sqrt(e_eff)) % y axis
            nx = x_dis; %cells along x = nx-1,Number of columns of Exc matrix
            ny = y_dis; %cells along y = ny-1 %Number of rows in Exc Matrix
            gndLength = 2*L;
            gndWidth = 2*W;  %pcbTraceWidth = 0.662e-3; % for 0.305 ro4003
            pcbTraceWidth = ((2*0.988e-3)/sqrt(e_eff))
            pcbTraceLength = 0.5*(gndLength - L);
            x = linspace(-L/2,L/2,nx); %bottom left coordinates of each cell
            y = linspace(-W/2,W/2,ny); %bottom left coordinates of each cell
            dx = x(2) - x(1); px = 1.25*dx % 5% bigger in size(for fabrication purpose)
            dy = y(2) - y(1); py = 1.25*dy
            first_one = true;
            feed = antenna.Rectangle('Length',2e-3,'Width',pcbTraceWidth,...
                'Center',[-L/2-1e-3,0]);
    
            Exc = ant_des;  
        
     for m=1:nx-1
        for n=1:ny-1
            if Exc(n,m) ~= 0 %only add these patches (includes -1 and 1)
                if first_one
                cboard = antenna.Rectangle('Length',px,'width',...
                py,'Center',[x(m)+dx/2,y(n)+dy/2]);
                first_one = false;
                else
                cboard = cboard + antenna.Rectangle('Length',px,'width',...
                py,'Center',[x(m)+dx/2,y(n)+dy/2]);
                end
            end
        end
    end
    % the feedsection
      cfed = cboard+feed;
      thick = 0.61e-3; %thickness
      p = pcbStack;
      d = dielectric('Name','Air','EpsilonR',1,...
            'Thickness',thick);
      gnd = antenna.Rectangle('Length',gndLength,'Width',gndWidth);
      gndPlane = gnd;
      p.BoardShape = gnd;
      p.BoardThickness = thick;
      p.Layers = {cfed,d,gnd};
      p.FeedLocations = [-L/2-1e-3,0,1,3];
      p.FeedDiameter = pcbTraceWidth/2-0.2e-3;
      p.Conductor = metal('copper');
      p.Conductor.Thickness = 35e-6;
      p.FeedVoltage = 1.0;
      p.FeedPhase = 0.0;
      show(p)
end