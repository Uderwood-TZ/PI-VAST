

clear; clc; close all;


beta  = 1.0;       % fixed propagation constant
D     = 4.0;       % transverse diffraction scale; D>1 broadens the soliton core physically
q     = 1;         % topological charge; q=1 gives a vortex ring and a 2*pi phase winding
L     = 6.0;       % square computational half-width: Omega=[-L,L]^2
Nmain = 201;       % main 2D grid size
gridList = [101, 151, 201];

Rrad = sqrt(2)*L;  % radial domain covers the square corners
NrMain = 1600;     % radial grid for main solution
maxIter = 8000;
tol = 1.0e-11;
relax = 0.70;

outdir = pwd;
figFormat = 'png';

fprintf('========== 2D stationary Kerr vortex soliton ==========' ); fprintf('\n');
fprintf('beta = %.6g, D = %.6g, q = %d, L = %.6g\n', beta, D, q, L);
fprintf('Main grid: %d x %d\n\n', Nmain, Nmain);

%% -------------------- Main simulation --------------------
fprintf('Solving radial vortex amplitude by FDM-Petviashvili...\n');
[r, f, info] = solve_radial_vortex(beta, D, q, Rrad, NrMain, maxIter, tol, relax);
fprintf('Converged: iter = %d, relative change = %.3e, S = %.12f\n', info.iter, info.relChange, info.S);
fprintf('Radial peak: max(f) = %.8f at r = %.8f\n\n', max(f), r(find(f==max(f),1,'first')));

fprintf('Reconstructing 2D complex field and derived fields...\n');
fields = make_2d_fields(r, f, beta, D, q, L, Nmain);

fprintf('Saving all field data and figures...\n');
save_all_outputs(fields, beta, D, q, outdir, figFormat);

fprintf('Computing grid-independence test...\n');
GI = grid_independence_test(beta, D, q, L, Rrad, gridList, maxIter, tol, relax);
writematrix(GI.table, fullfile(outdir, 'Kerr2D_grid_independence_table.csv'));
writecell(GI.header, fullfile(outdir, 'Kerr2D_grid_independence_header.csv'));
plot_grid_independence(GI, outdir, figFormat);

fprintf('Computing observed-order test with finest grid as reference...\n');
OO = observed_order_test(GI);
writematrix(OO.table, fullfile(outdir, 'Kerr2D_observed_order_table.csv'));
writecell(OO.header, fullfile(outdir, 'Kerr2D_observed_order_header.csv'));
plot_observed_order(OO, outdir, figFormat);

fprintf('Computing FDM Laplacian manufactured-order test...\n');
MMS = manufactured_order_test(beta, D, L, gridList);
writematrix(MMS.table, fullfile(outdir, 'Kerr2D_FDM_Laplacian_order_table.csv'));
writecell(MMS.header, fullfile(outdir, 'Kerr2D_FDM_Laplacian_order_header.csv'));
plot_mms_order(MMS, outdir, figFormat);

fprintf('\n========== Finished ==========' ); fprintf('\n');
fprintf('All outputs are saved in: %s\n', outdir);



function [r, f, info] = solve_radial_vortex(beta, D, q, Rrad, Nr, maxIter, tol, relax)
    r = linspace(0, Rrad, Nr).';
    dr = r(2) - r(1);
    nu = Nr - 1;                 % unknowns f(2:Nr), f(1)=0
    A = spalloc(nu, nu, 3*nu + 8);

    % Interior PDE rows: beta*f - (D/2)*(f_rr + r^{-1}f_r - q^2 r^{-2}f) = f^3
    for i = 2:Nr-1
        row = i - 1;
        ri = r(i);
        cim = 1/dr^2 - 1/(2*ri*dr);
        ci0 = -2/dr^2 - q^2/(ri^2);
        cip = 1/dr^2 + 1/(2*ri*dr);

        if i-1 >= 2
            col = (i-1) - 1;
            A(row, col) = A(row, col) - (D/2)*cim;
        end
        col = i - 1;
        A(row, col) = A(row, col) + beta - (D/2)*ci0;

        col = (i+1) - 1;
        A(row, col) = A(row, col) - (D/2)*cip;
    end

    % Outer Robin boundary: f_r(R) + sqrt(2*beta/D)*f(R) = 0
    lambda = sqrt(2*beta/D);
    row = nu;
    A(row, nu)   =  3/(2*dr) + lambda;
    A(row, nu-1) = -4/(2*dr);
    A(row, nu-2) =  1/(2*dr);

    A = sparse(A);
    try
        Afact = decomposition(A, 'lu');
        useDecomp = true;
    catch
        useDecomp = false;
    end

    width = 1.60*sqrt(D);
    f0 = 2.0*(r/width).^q .* exp(-0.5*(r/width).^2);
    f0(1) = 0;
    U = f0(2:Nr);

    gamma = 3/2;
    S = 1;
    relChange = inf;

    for it = 1:maxIter
        fFull = [0; U];
        [Nval, Kval, Mval] = radial_integrals(r, fFull, beta, D, q);
        S = (beta*Nval + 0.5*Kval)/(Mval + eps);

        rhs = U.^3;
        rhs(end) = 0;       % boundary equation row

        if useDecomp
            Uraw = Afact \ rhs;
        else
            Uraw = A \ rhs;
        end
        Unew = (S^gamma)*Uraw;

        relChange = norm(Unew - U, 2)/(norm(U, 2) + eps);
        U = relax*Unew + (1-relax)*U;

        if relChange < tol && abs(S - 1) < 5e-9
            break;
        end
    end

    f = [0; U];
    f = real(f);
    f(f < 0) = 0;

    info.iter = it;
    info.relChange = relChange;
    info.S = S;
end

function [Nval, Kval, Mval] = radial_integrals(r, f, beta, D, q)
    %#ok<INUSD>
    dr = r(2) - r(1);
    fr = gradient(f, dr);
    angular = zeros(size(f));
    angular(2:end) = q^2*f(2:end).^2 ./ (r(2:end).^2);
    angular(1) = q^2*fr(1)^2;

    Nval = 2*pi*trapz(r, f.^2 .* r);
    Kval = 2*pi*trapz(r, D*(fr.^2 + angular).*r);
    Mval = 2*pi*trapz(r, f.^4 .* r);
end

function fields = make_2d_fields(r, f, beta, D, q, L, N)
    x = linspace(-L, L, N);
    y = linspace(-L, L, N);
    dx = x(2) - x(1);
    dy = y(2) - y(1);
    [X, Y] = meshgrid(x, y);
    RR = sqrt(X.^2 + Y.^2);
    TH = atan2(Y, X);

    F = interp1(r, f, RR, 'pchip', 0);
    phi = F .* exp(1i*q*TH);
    phi(RR < 0.5*min(dx,dy)) = 0;

    absPhi = abs(phi);
    intensity = absPhi.^2;
    phaseField = angle(phi);

    [phi_y, phi_x] = gradient(phi, dy, dx);
    gradSq = abs(phi_x).^2 + abs(phi_y).^2;

    nDensity = intensity;
    kDensity = D*gradSq;
    mDensity = intensity.^2;
    enl = 0.5*mDensity;
    hDensity = 0.5*kDensity - 0.5*mDensity;
    etaDensity = abs(mDensity - 0.5*kDensity - beta*nDensity);

    Ru = 0.5*D*real(lap2(phi, dx, dy)) + intensity.*real(phi) - beta*real(phi);
    Rv = 0.5*D*imag(lap2(phi, dx, dy)) + intensity.*imag(phi) - beta*imag(phi);
    residualAbs = sqrt(Ru.^2 + Rv.^2);

    Nval = trapz(y, trapz(x, nDensity, 2));
    Kval = trapz(y, trapz(x, kDensity, 2));
    Mval = trapz(y, trapz(x, mDensity, 2));
    Hval = 0.5*Kval - 0.5*Mval;
    identityError = abs(Mval - 0.5*Kval - beta*Nval);

    [~, iy0] = min(abs(y));
    [~, ix0] = min(abs(x));
    abs_x0 = absPhi(:, ix0);
    abs_y0 = absPhi(iy0, :).';

    fields.x = x; fields.y = y; fields.X = X; fields.Y = Y;
    fields.phi = phi;
    fields.absPhi = absPhi;
    fields.intensity = intensity;
    fields.phase = phaseField;
    fields.enl = enl;
    fields.h = hDensity;
    fields.nDensity = nDensity;
    fields.kDensity = kDensity;
    fields.mDensity = mDensity;
    fields.etaDensity = etaDensity;
    fields.residualAbs = residualAbs;
    fields.abs_phi_x_0 = abs_y0;
    fields.abs_phi_0_y = abs_x0;
    fields.Nval = Nval;
    fields.Kval = Kval;
    fields.Mval = Mval;
    fields.Hval = Hval;
    fields.identityError = identityError;
    fields.beta = beta;
    fields.D = D;
    fields.q = q;
    fields.L = L;
    fields.Ngrid = N;
end

function Lphi = lap2(phi, dx, dy)
    Lphi = zeros(size(phi));
    Lphi(2:end-1, 2:end-1) = ...
        (phi(2:end-1, 3:end) - 2*phi(2:end-1, 2:end-1) + phi(2:end-1, 1:end-2))/dx^2 + ...
        (phi(3:end, 2:end-1) - 2*phi(2:end-1, 2:end-1) + phi(1:end-2, 2:end-1))/dy^2;
    Lphi(1,:) = Lphi(2,:);
    Lphi(end,:) = Lphi(end-1,:);
    Lphi(:,1) = Lphi(:,2);
    Lphi(:,end) = Lphi(:,end-1);
end

function save_all_outputs(fields, beta, D, q, outdir, figFormat)
    fieldList = {
        'abs_phi',              fields.absPhi,      '|phi(x,y)|';
        'abs_phi_squared',      fields.intensity,   '|phi(x,y)|^2';
        'arg_phi',              fields.phase,       'arg phi(x,y)';
        'e_nl',                 fields.enl,         'e_{nl}(x,y)';
        'h_density',            fields.h,           'h(x,y)';
        'n_density',            fields.nDensity,    'n(x,y)=|phi|^2';
        'k_density',            fields.kDensity,    'k(x,y)=D|grad phi|^2';
        'm_density',            fields.mDensity,    'm(x,y)=|phi|^4';
        'identity_density_eta', fields.etaDensity,  '|m-k/2-beta n|';
        'pde_residual_abs',     fields.residualAbs, '|R(x,y)|'
    };

    for j = 1:size(fieldList,1)
        name = fieldList{j,1};
        val  = fieldList{j,2};
        ttl  = fieldList{j,3};
        save_field_xyz(fullfile(outdir, ['Kerr2D_' name '.csv']), fields.X, fields.Y, val);
        plot_field(fields.x, fields.y, val, ttl, fullfile(outdir, ['Kerr2D_' name '.' figFormat]));
    end

    x = fields.x(:);
    y = fields.y(:);
    save_line_xyz(fullfile(outdir, 'Kerr2D_abs_phi_x_0.csv'), x, zeros(size(x)), fields.abs_phi_x_0(:));
    save_line_xyz(fullfile(outdir, 'Kerr2D_abs_phi_0_y.csv'), zeros(size(y)), y, fields.abs_phi_0_y(:));

    plot_line(x, fields.abs_phi_x_0(:), 'x', '|phi(x,0)|', fullfile(outdir, ['Kerr2D_abs_phi_x_0.' figFormat]));
    plot_line(y, fields.abs_phi_0_y(:), 'y', '|phi(0,y)|', fullfile(outdir, ['Kerr2D_abs_phi_0_y.' figFormat]));

    globalHeader = {'beta','D','q','N','K','M','H','abs_M_minus_K_over_2_minus_beta_N'};
    globalTable = [beta, D, q, fields.Nval, fields.Kval, fields.Mval, fields.Hval, fields.identityError];
    writecell(globalHeader, fullfile(outdir, 'Kerr2D_global_quantities_header.csv'));
    writematrix(globalTable, fullfile(outdir, 'Kerr2D_global_quantities.csv'));

    figure('Color','w');
    bar([fields.Nval, fields.Kval, fields.Mval, fields.Hval, fields.identityError]);
    set(gca, 'XTickLabel', {'N','K','M','H','|M-K/2-betaN|'});
    ylabel('value');
    title('Global integral quantities');
    grid on;
    save_figure(fullfile(outdir, ['Kerr2D_global_quantities.' figFormat]));
end

function save_field_xyz(filename, X, Y, V)
    writematrix([X(:), Y(:), V(:)], filename);
end

function save_line_xyz(filename, X, Y, V)
    writematrix([X(:), Y(:), V(:)], filename);
end

function plot_field(x, y, V, ttl, filename)
    figure('Color','w');
    imagesc(x, y, V);
    set(gca, 'YDir', 'normal');
    axis image;
    xlabel('x'); ylabel('y');
    title(ttl, 'Interpreter','tex');
    colorbar;
    save_figure(filename);
end

function plot_line(x, v, xlab, ylab, filename)
    figure('Color','w');
    plot(x, v, 'LineWidth', 1.8);
    xlabel(xlab); ylabel(ylab);
    title(ylab, 'Interpreter','tex');
    grid on;
    save_figure(filename);
end

function save_figure(filename)
    try
        exportgraphics(gcf, filename, 'Resolution', 300);
    catch
        saveas(gcf, filename);
    end
    close(gcf);
end

function GI = grid_independence_test(beta, D, q, L, Rrad, gridList, maxIter, tol, relax)
    ng = numel(gridList);
    sol = cell(ng,1);
    table = zeros(ng, 10);

    for a = 1:ng
        N = gridList(a);
        Nr = max(900, 6*N);
        [r, f, info] = solve_radial_vortex(beta, D, q, Rrad, Nr, maxIter, tol, relax);
        F = make_2d_fields(r, f, beta, D, q, L, N);
        sol{a} = F;
        dx = 2*L/(N-1);
        table(a,:) = [N, dx, info.iter, info.relChange, F.Nval, F.Kval, F.Mval, F.Hval, F.identityError, max(F.absPhi(:))];
    end

    GI.header = {'Ngrid','dx','iteration','relative_change','N','K','M','H','abs_M_minus_K_over_2_minus_beta_N','max_abs_phi'};
    GI.table = table;
    GI.sol = sol;
end

function plot_grid_independence(GI, outdir, figFormat)
    T = GI.table;
    figure('Color','w');
    plot(T(:,2), T(:,5), '-o', 'LineWidth', 1.6); hold on;
    plot(T(:,2), T(:,6), '-s', 'LineWidth', 1.6);
    plot(T(:,2), T(:,7), '-^', 'LineWidth', 1.6);
    set(gca, 'XDir', 'reverse');
    xlabel('grid spacing dx'); ylabel('integral value');
    legend('N','K','M','Location','best');
    title('Grid-independence test for N, K, M');
    grid on;
    save_figure(fullfile(outdir, ['Kerr2D_grid_independence_NKM.' figFormat]));

    figure('Color','w');
    semilogy(T(:,2), T(:,9), '-o', 'LineWidth', 1.6);
    set(gca, 'XDir', 'reverse');
    xlabel('grid spacing dx'); ylabel('|M-K/2-beta N|');
    title('Grid-independence test for global identity residual');
    grid on;
    save_figure(fullfile(outdir, ['Kerr2D_grid_independence_identity.' figFormat]));
end

function OO = observed_order_test(GI)
    sol = GI.sol;
    ng = numel(sol);
    ref = sol{ng};
    xRef = ref.x;
    yRef = ref.y;
    absRef = ref.absPhi;
    h = GI.table(:,2);

    err = zeros(ng-1,1);
    for a = 1:ng-1
        S = sol{a};
        [Xq, Yq] = meshgrid(xRef, yRef);
        absInterp = interp2(S.x, S.y, S.absPhi, Xq, Yq, 'spline');
        diff = absInterp - absRef;
        err(a) = sqrt(mean(diff(:).^2));
    end

    p = nan(ng-2,1);
    for a = 1:ng-2
        p(a) = log(err(a)/err(a+1))/log(h(a)/h(a+1));
    end

    maxRows = max(numel(err), numel(p));
    table = nan(maxRows, 4);
    table(1:numel(err),1) = GI.table(1:end-1,1);
    table(1:numel(err),2) = h(1:end-1);
    table(1:numel(err),3) = err;
    table(1:numel(p),4) = p;

    OO.header = {'Ngrid','dx','RMSE_abs_phi_against_finest','observed_order'};
    OO.table = table;
end

function plot_observed_order(OO, outdir, figFormat)
    T = OO.table;
    validErr = ~isnan(T(:,3));
    figure('Color','w');
    loglog(T(validErr,2), T(validErr,3), '-o', 'LineWidth', 1.6);
    xlabel('grid spacing dx'); ylabel('RMSE of |phi|');
    title('Observed grid-refinement error');
    grid on;
    save_figure(fullfile(outdir, ['Kerr2D_observed_grid_error.' figFormat]));

    validP = ~isnan(T(:,4));
    figure('Color','w');
    plot(T(validP,2), T(validP,4), '-o', 'LineWidth', 1.6); hold on;
    yline(2, '--', 'Second order reference');
    xlabel('coarse-grid dx'); ylabel('observed order');
    title('Observed order test');
    grid on;
    save_figure(fullfile(outdir, ['Kerr2D_observed_order.' figFormat]));
end

function MMS = manufactured_order_test(beta, D, L, gridList)
    ng = numel(gridList);
    table = zeros(ng, 4);
    a = 0.12;

    for k = 1:ng
        N = gridList(k);
        x = linspace(-L, L, N);
        y = linspace(-L, L, N);
        dx = x(2) - x(1);
        dy = y(2) - y(1);
        [X, Y] = meshgrid(x, y);
        R2 = X.^2 + Y.^2;
        phi = exp(-a*R2);

        lapExact = (-4*a + 4*a^2*R2).*phi;
        lapNum = lap2(phi, dx, dy);
        errMat = lapNum(2:end-1, 2:end-1) - lapExact(2:end-1, 2:end-1);
        rmse = sqrt(mean(errMat(:).^2));
        table(k,1:3) = [N, dx, rmse];
    end

    for k = 1:ng-1
        table(k,4) = log(table(k,3)/table(k+1,3))/log(table(k,2)/table(k+1,2));
    end
    table(ng,4) = NaN;

    MMS.header = {'Ngrid','dx','RMSE_Laplacian','observed_order'};
    MMS.table = table;
end

function plot_mms_order(MMS, outdir, figFormat)
    T = MMS.table;
    figure('Color','w');
    loglog(T(:,2), T(:,3), '-o', 'LineWidth', 1.6); hold on;
    ref = T(1,3)*(T(:,2)/T(1,2)).^2;
    loglog(T(:,2), ref, '--', 'LineWidth', 1.2);
    xlabel('grid spacing dx'); ylabel('RMSE');
    legend('FDM Laplacian error','O(dx^2) reference','Location','best');
    title('Manufactured-order test for the FDM Laplacian');
    grid on;
    save_figure(fullfile(outdir, ['Kerr2D_FDM_Laplacian_order.' figFormat]));
end
