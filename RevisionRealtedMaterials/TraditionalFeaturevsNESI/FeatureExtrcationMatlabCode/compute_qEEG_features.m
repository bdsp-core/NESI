function [SEF95,RelativeDelta,DeltaAlpha]=compute_qEEG_features(EEG,Fs)

% EEG: N_cases x N_channels x N_samples
% Fs : sampling frequency (e.g., 200 Hz)
%
% Outputs:
% SEF95        : N_cases x N_channels
% RelativeDelta: N_cases x N_channels
% DeltaAlpha   : N_cases x N_channels

if nargin<2
    Fs=200;
end

N_cases=size(EEG,1);
N_channels=size(EEG,2);

SEF95=NaN(N_cases,N_channels);
RelativeDelta=NaN(N_cases,N_channels);
DeltaAlpha=NaN(N_cases,N_channels);

for n=1:N_cases
    fprintf('Processing case %d/%d\n',n,N_cases);

    for ch=1:N_channels

        x=squeeze(EEG(n,ch,:));
        x=double(x(:));

        if all(isnan(x)) || sum(~isnan(x))<Fs
            continue;
        end

        % Fill occasional NaNs
        x=fillmissing(x,'linear','EndValues','nearest');

        % Same band definitions as fcn_computeFeatures_powers_complex_v2
        P1=bandpower(x,Fs,[0.5 4]);   % Delta
        P2=bandpower(x,Fs,[4 7]);     % Theta
        P3=bandpower(x,Fs,[8 15]);    % Alpha
        P4=bandpower(x,Fs,[16 31]);   % Beta
        P5=bandpower(x,Fs,[32 64]);   % Gamma

        P=P1+P2+P3+P4+P5;

        % Relative delta: exactly same concept as original BDSP function
        RelativeDelta(n,ch)=P1/(P+eps);

        % Delta/alpha: exactly same concept as original BDSP function
        DeltaAlpha(n,ch)=P1/(P3+eps);

        % -------------------------------------------------------------
        % SEF95
        % Calculate PSD using Welch's method
        % -------------------------------------------------------------
        window_length=4*Fs;       % 4-second window
        overlap=round(0.5*window_length);
        nfft=max(2048,2^nextpow2(window_length));

        [pxx,f]=pwelch(x,hamming(window_length),overlap,nfft,Fs);

        % Use same overall spectral range as the band-power features
        idx=(f>=0.5 & f<=64);
        pxx_use=pxx(idx);
        f_use=f(idx);

        % Cumulative spectral power
        cumulative_power=cumtrapz(f_use,pxx_use);
        total_power=cumulative_power(end);

        if total_power>0
            target=0.95*total_power;
            k=find(cumulative_power>=target,1,'first');

            % Linear interpolation for more precise SEF95
            if k>1
                p1=cumulative_power(k-1);
                p2=cumulative_power(k);
                f1=f_use(k-1);
                f2=f_use(k);

                if p2>p1
                    SEF95(n,ch)=f1+(target-p1)*(f2-f1)/(p2-p1);
                else
                    SEF95(n,ch)=f_use(k);
                end
            else
                SEF95(n,ch)=f_use(k);
            end
        end
    end
end
end