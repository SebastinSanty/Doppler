import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import spline


def prepareData(rawfilename, f1, f2):
    noise_multiplier = 1.0
    mean_cutoff = False
    x_axis_unit = 20
    exponentiate = False
    silence = True
    smoothing_factor=100
    plot_with_noise=True


    data = np.loadtxt(rawfilename)

    noise_cutoff = noise_multiplier * np.median(data, axis=0)[2]
    if (mean_cutoff):
        noise_cutoff = noise_multiplier * np.mean(data, axis=0)[2]

    if exponentiate:
        noise_cutoff = np.exp(noise_cutoff)

    df = pd.DataFrame(data, columns=['time', 'freq', 'amp'])

    if exponentiate:
        df['amp'] = np.exp(df['amp'])

    df.loc[df['amp']<noise_cutoff, 'amp'] = 0

    if silence:#WHY?
        df.loc[(df['freq']<600) & (df['freq']>500), 'amp'] = 0
        df.loc[(df['freq']<700) & (df['freq']>650), 'amp'] = 0
        df.loc[(df['freq']<885) & (df['freq']>800), 'amp'] = 0


    if plot_with_noise:
        df = pd.DataFrame(data, columns=['time', 'freq', 'amp'])

    sdf = df.drop(df[df.freq < f1-10].index)
    sdf = sdf.drop(sdf[sdf.freq > f1+10].index)
    sdf = sdf.drop(sdf[sdf.amp == 0].index)

    sdf2 = df.drop(df[df.freq < f2-10].index)
    sdf2 = sdf2.drop(sdf[sdf.freq > f2+10].index)
    sdf2 = sdf2.drop(sdf[sdf.amp == 0].index)

    sdf['amp'] = sdf['amp'].add(other = sdf2['amp'],fill_value = 0)

    graph = sdf.groupby(['time'], sort=False)['amp'].max()
    freq = sdf.groupby(['time'], sort=False)['freq'].max()

    graph = pd.rolling_mean(graph, smoothing_factor, center=False)
    
    return freq, graph


def preproc(graph1, graph2, freq1, freq2):
    graph3 = pd.concat([graph1, freq1, graph2, freq2], axis=1)
    graph3.columns = ['amp1', 'freq1', 'amp2', 'freq2']
    graph3 = graph3[np.isfinite(graph3['amp1'])]
    graph3 = graph3[np.isfinite(graph3['amp2'])]
    return graph3