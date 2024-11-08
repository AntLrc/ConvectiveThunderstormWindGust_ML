from vgam_loader import RExperiment
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
expnumbers = [10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
result = []
baseLogLik = RExperiment(f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/R_experiments/experiment_29.pkl").LogLik['values']
for i in expnumbers:
    exp = RExperiment(f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/R_experiments/experiment_{i}.pkl")
    for ifold, lmle in enumerate(exp.LogLik['values']):
        if ifold != 0:
            continue
        result.append([exp.features[0], lmle, 1 - lmle/baseLogLik[ifold], ifold])
df = pd.DataFrame(data = [[k[0], k[1], k[2], k[3]] for k in result], columns = ['Variable', 'LogLik', 'Skill score', 'Fold'])
df = df.sort_values(by = 'Skill score', ascending = False)
sns.catplot(data = df, x = "Variable", y = "Skill score", hue = "Variable", row = 'Fold', kind = 'bar', palette = "rocket", sharex = False)
# Rotate all x labels
for ax in plt.gcf().axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
plt.title("Sequential feature selection: step 3")
plt.tight_layout()
plt.savefig("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/DiagnosticsVal/SeqFeatSel3.png")
with open("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/DiagnosticsVal/SeqFeatSel3.pkl", "wb") as f:
    pickle.dump(df, f)
plt.close()