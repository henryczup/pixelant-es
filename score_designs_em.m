function score_designs_em(input_mat, output_mat)
%SCORE_DESIGNS_EM Headless MATLAB EM scorer for 12x12 binary masks.
%
% Input .mat fields:
%   designs      Numeric/logical masks shaped [N,12,12], [12,12,N], or [12,12]
%   freq         Optional frequency vector. Defaults depend on solver_mode.
%   solver_mode  "air" or "substrate". Default "air".
%   options      Optional struct with mesh_divisor, impedance, scale_factor,
%                epsilon_r, loss_tangent, thickness.
%
% Output .mat fields:
%   s11_db          [N, numel(freq)] spectra. Failed rows are NaN.
%   valid           [N,1] logical success flags.
%   error_message   [N,1] cell array with per-design error text.
%   elapsed_seconds [N,1] solver wall time per design.
%   freq            Frequency vector used.
%   solver_mode     Solver mode used.

    in = load(input_mat);
    if ~isfield(in, 'designs')
        error('score_designs_em:MissingDesigns', 'Input mat must contain a designs variable.');
    end

    solver_mode = "air";
    if isfield(in, 'solver_mode')
        solver_mode = string(in.solver_mode);
    end

    options = struct();
    if isfield(in, 'options')
        options = in.options;
    end

    freq = default_frequency(solver_mode);
    if isfield(in, 'freq') && ~isempty(in.freq)
        freq = double(in.freq(:)).';
    end

    designs = normalize_designs(in.designs);
    n = size(designs, 1);
    s11_db = nan(n, numel(freq));
    valid = false(n, 1);
    error_message = cell(n, 1);
    elapsed_seconds = nan(n, 1);
    impedance = get_option(options, 'impedance', 50);
    mesh_divisor = get_option(options, 'mesh_divisor', 15);
    lambda = 3e8 / max(freq);

    for ii = 1:n
        t_start = tic;
        try
            design = squeeze(designs(ii, :, :));
            design = enforce_feed(design);
            p = build_pixelated_pcb(design, solver_mode, options);
            mesh(p, 'MaxEdgeLength', lambda / mesh_divisor);
            s = sparameters(p, freq, impedance);
            s11 = rfparam(s, 1, 1);
            s11_db(ii, :) = 20 * log10(abs(s11(:))).';
            valid(ii) = all(isfinite(s11_db(ii, :)));
            if ~valid(ii)
                error_message{ii} = 'Non-finite S11 values returned by solver.';
            else
                error_message{ii} = '';
            end
        catch ME
            valid(ii) = false;
            error_message{ii} = ME.message;
        end
        elapsed_seconds(ii) = toc(t_start);
    end

    save(output_mat, 's11_db', 'valid', 'error_message', 'elapsed_seconds', 'freq', 'solver_mode');
end

function designs = normalize_designs(raw)
    raw = double(raw);
    dims = size(raw);
    if isequal(dims, [12, 12])
        designs = reshape(raw, [1, 12, 12]);
    elseif ndims(raw) == 3 && dims(2) == 12 && dims(3) == 12
        designs = raw;
    elseif ndims(raw) == 3 && dims(1) == 12 && dims(2) == 12
        designs = permute(raw, [3, 1, 2]);
    else
        error('score_designs_em:BadShape', 'Expected designs shaped [N,12,12], [12,12,N], or [12,12].');
    end
    designs = designs > 0.5;
end

function design = enforce_feed(design)
    design(6:7, 1) = 1;
end

function freq = default_frequency(solver_mode)
    if solver_mode == "air"
        freq = linspace(10e9, 20e9, 81);
    elseif solver_mode == "substrate"
        freq = linspace(1e9, 5e9, 81);
    else
        error('score_designs_em:BadMode', 'solver_mode must be air or substrate.');
    end
end

function value = get_option(options, name, default_value)
    if isfield(options, name) && ~isempty(options.(name))
        value = options.(name);
    else
        value = default_value;
    end
end

function p = build_pixelated_pcb(ant_des, solver_mode, options)
    x_dis = 13;
    y_dis = 13;
    nx = x_dis;
    ny = y_dis;

    if solver_mode == "air"
        e_eff = 1;
        scale_factor = get_option(options, 'scale_factor', 1);
        dielectric_name = 'Air';
        epsilon_r = get_option(options, 'epsilon_r', 1);
        loss_tangent = [];
        thick = get_option(options, 'thickness', 0.61e-3);
    elseif solver_mode == "substrate"
        e_eff = get_option(options, 'e_eff', 4.2516);
        scale_factor = get_option(options, 'scale_factor', 4);
        dielectric_name = 'FR4';
        epsilon_r = get_option(options, 'epsilon_r', 4.8);
        loss_tangent = get_option(options, 'loss_tangent', 0.0260);
        thick = get_option(options, 'thickness', 3.2e-3);
    else
        error('score_designs_em:BadMode', 'solver_mode must be air or substrate.');
    end

    L = scale_factor * ((2 * 3.75e-3) / sqrt(e_eff));
    W = scale_factor * ((2 * 3.75e-3) / sqrt(e_eff));
    gndLength = 2 * L;
    gndWidth = 2 * W;
    pcbTraceWidth = ((2 * 0.988e-3) / sqrt(e_eff));
    x = linspace(-L / 2, L / 2, nx);
    y = linspace(-W / 2, W / 2, ny);
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    px = 1.25 * dx;
    py = 1.25 * dy;

    feed = antenna.Rectangle('Length', 2e-3, 'Width', pcbTraceWidth, 'Center', [-L / 2 - 1e-3, 0]);
    first_one = true;
    for m = 1:nx-1
        for n = 1:ny-1
            if ant_des(n, m) ~= 0
                rect = antenna.Rectangle('Length', px, 'Width', py, 'Center', [x(m) + dx / 2, y(n) + dy / 2]);
                if first_one
                    cboard = rect;
                    first_one = false;
                else
                    cboard = cboard + rect;
                end
            end
        end
    end

    if first_one
        cboard = antenna.Rectangle('Length', px, 'Width', py, 'Center', [x(1) + dx / 2, y(1) + dy / 2]);
    end

    cfed = cboard + feed;
    gnd = antenna.Rectangle('Length', gndLength, 'Width', gndWidth);
    p = pcbStack;
    if solver_mode == "air"
        d = dielectric('Name', dielectric_name, 'EpsilonR', epsilon_r, 'Thickness', thick);
    else
        d = dielectric('Name', dielectric_name, 'EpsilonR', epsilon_r, 'LossTangent', loss_tangent, 'Thickness', thick);
    end
    p.BoardShape = gnd;
    p.BoardThickness = thick;
    p.Layers = {cfed, d, gnd};
    p.FeedLocations = [-L / 2 - 1e-3, 0, 1, 3];
    p.FeedDiameter = pcbTraceWidth / 2 - 0.2e-3;
    p.Conductor = metal('copper');
    p.Conductor.Thickness = 35e-6;
    p.FeedVoltage = 1.0;
    p.FeedPhase = 0.0;
end
