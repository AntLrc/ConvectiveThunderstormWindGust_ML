from vgam_loader import RExperiment
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
expnumbers = [12, 14, 15, 17, 19, 20, 21, 24, 25, 27, 28]
result = []
baseLogLik = RExperiment(f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/R_experiments/experiment_29.pkl").LogLik['mean']
for i in expnumbers:
    exp = RExperiment(f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/test/R_experiments/experiment_{i}.pkl")
    for lmle in exp.LogLik['values']:
        result.append([exp.features[0], lmle])
df = pd.DataFrame(data = [[k[0], k[1]] for k in result], columns = ['Variable', 'LogLik'])
df = pd.DataFrame(data = [[k[0], k[1]] for k in result], columns = ['Variable', 'LogLik'])
df['Skill score'] = 1- df.LogLik/baseLogLik
df = df.sort_values(by = 'Skill score', ascending = False)
order = df.groupby('Variable').mean().sort_values('Skill score').index[::-1]
sns.barplot(data = df, x = "Variable", y = "Skill score", hue = "Variable", palette = "rocket", order = order)
plt.xticks(rotation = 90)
plt.title("Sequential feature selection: step 8")
plt.tight_layout()
with open("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/Diagnostics/SeqFeatSel7.pkl", "rb") as f:
    df0 = pickle.load(f)
x = df0.where(df0['Variable'] == 'q_1000hPa')['Skill score'].dropna().mean()
values = [x]*len(order)
plt.plot(order, values, '--')
plt.savefig("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/Diagnostics/SeqFeatSel8.png")
with open("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/Diagnostics/SeqFeatSel8.pkl", "wb") as f:
    pickle.dump(df, f)
plt.close()
