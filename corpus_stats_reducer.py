import glob
import numpy as np

save_dir = 'data/'

print("Summing Document Frequencies")
all_dfs = []
all_Ns = []
for filename in glob.glob(save_dir + '/doc*'):
    N = int(re.findall('N\d+', filename)[0][1:])
    all_Ns.append(N)    
    df = np.load(filename)
    all_dfs.append(df)

df_final = np.vstack(all_dfs).sum(axis=0)
N_final = sum(all_Ns)
np.save(save_dir + '/final-N%s-dfs.npy' % N_final, df_final)

print("Calculating IDF")
idf = np.log((N_final+1) / (df_final+1))
np.save(save_dir + '/final-N%s-idf.npy' % N_final, maxes_final)

print("Summing Maxes")
all_maxes = []
for filename in glob.glob(save_dir + '/maxes*'):
    maxes = np.load(filename)
    all_maxes.append(maxes)
maxes_final = np.vstack(all_maxes).sum(axis=0)

np.save(save_dir + '/final-N%s-maxes.npy' % N_final, maxes_final)