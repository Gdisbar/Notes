#########################
# Augmentation methods
#########################
def noise(data):
    """
    Adding White Noise to background,not too much static such that it doesn't obfuscate the signal too much. 
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
    
def shift(data):
    """
    Random Shifting.So its not very noticable but what I've done there is move the audio randomly to either the left 
    or right direction, within the fix audio duration. So if you compare this to the original plot, you can see the same 
    audio wave pattern, except there's a tiny bit of delay before the speaker starts speaking. 
    """
    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500
    return np.roll(data, s_range)
    
def stretch(data, rate=0.8):
    """
    Streching the Sound. Note that this expands the dataset slightly. This one is one of the more dramatic augmentation methods. 
    The method literally stretches the audio. So the duration is longer, but the audio wave gets strecthed too. 
    Thus introducing and effect that sounds like a slow motion sound. If you look at the audio wave itself, you'll 
    notice that compared to the orginal audio, the strected audio seems to hit a higher frequency note. Thus creating a 
    more diverse data for augmentation. Pretty nifty eh? It does introduce abit of a 
    challenge in the data prep stage cause it lengthens the audio duration. Something to consider especially when 
    doing a 2D CNN
    """
    data = librosa.effects.time_stretch(data, rate)
    return data
    
def pitch(data, sample_rate):
    """
    Pitch Tuning, this method accentuates the high pitch notes, by... normalising it sort of.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data
    
def dyn_change(data):
    """
    Random Value Change.It's exactly the same as the original. Yes true, but if you look at the frequency, the wave hits higher 
    frequency notes compared to the original 
    where the min is around -1 and the max is around 1. The min and max of this audio is -6 and 6 respestively.
    """
    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3
    return (data * dyn_change)
    
def speedNpitch(data):
    """
    peed and Pitch Tuning.It compresses the audio wave but keeping the audio duration the same. If you listen to it, the 
    effect is opposite of the stretch augmentation method. An angry person when applied this augmentation method, 
    to the human ear, will really alter the emotion interpretation of this audio. Not sure if this is counter productive to 
    the algorithm, but lets try it. 
    Another potential, downside is that there will be silence in the later part of the audio. 
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

## Testing on single audio file
data,sr = librosa.load(path)
x = speedNpitch(data)
plt.figure(figsize=(15, 5))
librosa.display.waveshow(x, sr=sampling_rate)
Audio(x, rate=sampling_rate)

## ref -> labels,source,path

# Note this takes a couple of minutes (~16 mins) as we're iterating over 4 datasets, and with augmentation  
df = pd.DataFrame(columns=['feature'])
df_noise = pd.DataFrame(columns=['feature'])
df_speedpitch = pd.DataFrame(columns=['feature'])
cnt = 0
# loop feature extraction over the entire dataset
for i in tqdm(ref.path):
    # first load the audio 
    X, sample_rate = librosa.load(i, res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)
    # take mfcc and mean as the feature. Could do min and max etc as well. 
    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=np.array(sample_rate),n_mfcc=13),axis=0)
    df.loc[cnt] = [mfccs]   
    # noise 
    aug = noise(X)
    aug = np.mean(librosa.feature.mfcc(y=aug,sr=np.array(sample_rate),n_mfcc=13),axis=0)              
    df_noise.loc[cnt] = [aug]
    # speed pitch
    aug = speedNpitch(X)
    aug = np.mean(librosa.feature.mfcc(y=aug,sr=np.array(sample_rate), n_mfcc=13),axis=0)         
    df_speedpitch.loc[cnt] = [aug]   
    cnt += 1

    
    
df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)
df_noise = pd.concat([ref,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)
df_speedpitch = pd.concat([ref,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)
#print(df.shape,df_noise.shape,df_speedpitch.shape) (12162, 219) (12162, 219) (12162, 219)
# So new dataset once stacked on top, will be 3 times the original size, which is handy since Deep Learning needs alot of data

df = pd.concat([df,df_noise,df_speedpitch],axis=0,sort=False)  #  219 columns ->  labels,source,path & 216 mfcc
df=df.fillna(0)
# del df_noise, df_speedpitch

==============================================================================================================
==============================================================================================================
## Tested on Audio to transcript generation rresult is not good at all
==============================================================================================================
==============================================================================================================

def get_forground_audio(path):
    data,sr=librosa.load(path)
    S_full,phase=librosa.magphase(librosa.stft(data))
    S_filter=librosa.decompose.nn_filter(S_full,aggregate=np.median,metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    # The output of the filter should not be greater than the input if we consider the signals to be 
    # additive . Taking a point minimum forces this with the input spectrum.
    S_filter=np.minimum(S_full, S_filter)
    margin_bg,margin_fg,power=2,10,2
    mask_bg=librosa.util.softmask(S_filter,margin_bg*(S_full-S_filter),power=power)
    mask_fg=librosa.util.softmask(S_full-S_filter,margin_fg*S_filter,power=power)
    # After we get the masks, multiply them with the input spectrum to separate the components
    S_foreground=mask_fg*S_full
    # foreground_audio = librosa.amplitude_to_db(S_foreground[:,:], ref=np.max)
    foreground=librosa.istft(S_foreground * phase)
    
    return foreground
  
#reduced_noise = nr.reduce_noise(y=y_foreground, sr=sr)
y_foreground=get_forground_audio(path)
reduced_noise = nr.reduce_noise(y=y_foreground, sr=16000) # now this will mess up the sr won't be able to change to 16KHz
wavfile.write(path,sr,reduced_noise)


# test = AudioSegment.from_file(path)
# test.export(path) #Exports to a mp3 file in the current path.
# Audio(path, autoplay=False)


