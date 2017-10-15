import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

rawfilename = 'FrequencyEvents.txt'
noise_multiplier = 1.5
mean_cutoff = False
x_axis_unit = 20
exponentiate = False
silence = True
smoothing_factor=100
plot_with_noise=True


data = np.loadtxt(rawfilename)
print(type(data[1]))
print(data[1])
print()
print(data[1][0])
print(data[1][1])
print(data[1][2])

noise_cutoff = noise_multiplier * np.median(data, axis=0)[2]
if (mean_cutoff):
    noise_cutoff = noise_multiplier * np.mean(data, axis=0)[2]

if exponentiate:
    noise_cutoff = np.exp(noise_cutoff)

print()
print(np.median(data, axis=0)[2])
print(np.mean(data, axis=0)[2])
print(noise_cutoff)

df = pd.DataFrame(data, columns=['time', 'freq', 'amp'])

#print(df.loc[df['time']==0])

print()
print(df.head)

if exponentiate:
    df['amp'] = np.exp(df['amp'])

df.loc[df['amp']<noise_cutoff, 'amp'] = 0

if silence:
    df.loc[(df['freq']<600) & (df['freq']>500), 'amp'] = 0
    df.loc[(df['freq']<700) & (df['freq']>650), 'amp'] = 0
    df.loc[(df['freq']<885) & (df['freq']>800), 'amp'] = 0

print()
print(df.head)


# plots = df.hist('freq', weights=df['amp'], bins=1200)
# print(plt)

# ticks = np.arange(0, 1201, x_axis_unit)
# labels = [x_axis_unit*i for i in range(ticks.size)]
# plt.xticks(ticks, labels)
# plt.xlim(350, 1050)

# plt.show()
print('histogram printed')


if plot_with_noise:
    df = pd.DataFrame(data, columns=['time', 'freq', 'amp'])
    # not executing this line doesn't reread noisy data, hence preserving the df with noise cancelling








sdf = df.drop(df[df.freq < 705].index)
sdf = sdf.drop(sdf[sdf.freq > 725].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='715', x='time', y='freq')










sdf = df.drop(df[df.freq < 445].index)
sdf = sdf.drop(sdf[sdf.freq > 460].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='450', x='time', y='freq')







sdf = df.drop(df[df.freq < 422].index)
sdf = sdf.drop(sdf[sdf.freq > 435].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='430', x='time', y='freq')







sdf = df.drop(df[df.freq < 390].index)
sdf = sdf.drop(sdf[sdf.freq > 410].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='400', x='time', y='freq')






sdf = df.drop(df[df.freq < 610].index)
sdf = sdf.drop(sdf[sdf.freq > 630].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='620', x='time', y='freq')









sdf = df.drop(df[df.freq < 890].index)
sdf = sdf.drop(sdf[sdf.freq > 910].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='900', x='time', y='freq')











sdf = df.drop(df[df.freq < 910].index)
sdf = sdf.drop(sdf[sdf.freq > 925].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='917', x='time', y='freq')








sdf = df.drop(df[df.freq < 985].index)
sdf = sdf.drop(sdf[sdf.freq > 1010].index)
sdf = sdf.drop(sdf[sdf.amp == 0].index)

graph = sdf.groupby(['time'], sort=False)['amp'].max()

graph = pd.rolling_mean(graph, smoothing_factor, center=False)
graph.plot(label='1000', x='time', y='freq')







plt.legend(loc='upper left')
plt.show()


