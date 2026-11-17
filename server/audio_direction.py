"""GCC-PHAT for a calibrated two-microphone pair; no direction from mono."""
import math

import numpy as np


def direction(samples, sample_rate, spacing=None, array_yaw=0):
    if samples.ndim != 2 or samples.shape[1] != 2 or spacing is None:
        return {"available":False,"reason":"单声道或未标定麦克风间距，无法推断声源方向"}
    spacing = float(spacing)
    if not math.isfinite(spacing) or not .02 <= spacing <= .5:
        raise ValueError("麦克风间距必须在 0.02–0.5 米")
    left, right = samples[:,0], samples[:,1]
    if np.std(left-right) < 1e-5:
        return {"available":False,"reason":"两个声道相同；不是独立麦克风阵列信号"}
    if min(np.sqrt(np.mean(left**2)), np.sqrt(np.mean(right**2))) < .002:
        return {"available":False,"reason":"声道能量不足"}
    length = 1 << (2*len(left)-1).bit_length()
    cross = np.fft.rfft(left, length)*np.conj(np.fft.rfft(right, length))
    correlation = np.fft.irfft(cross/np.maximum(np.abs(cross),1e-12), length*8)
    max_shift = min(int(sample_rate*8*spacing/343), len(correlation)//2-1)
    bounded = np.r_[correlation[-max_shift:], correlation[:max_shift+1]]
    shift = int(np.argmax(np.abs(bounded)))-max_shift
    delay = shift/(sample_rate*8)
    # Pair positions: left=-spacing/2, right=+spacing/2. Positive is ego-right.
    azimuth = math.degrees(math.asin(np.clip(delay*343/spacing,-1,1)))
    peak = float(np.max(np.abs(bounded)))
    quality = peak/max(float(np.mean(np.abs(bounded))),1e-8)
    return {"available":quality >= 2, "azimuth":round(azimuth,2),
            "worldBearing":round(float(array_yaw)+azimuth,2), "delaySeconds":delay,
            "peakRatio":round(quality,2),"frontBackAmbiguous":True,
            "reason":"双麦仅提供左右角，存在前后镜像歧义；真实音源不得直接当作仿真世界位置"}
