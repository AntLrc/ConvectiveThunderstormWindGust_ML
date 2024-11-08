from vgam_loader import RExperiment
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
df = None
for i in range(1,8):
    with open("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/Diagnostics/SeqFeatSel"+str(i)+".pkl", "rb") as f:
        df0 = pickle.load(f)
    var = df0.groupby('Variable').mean().sort_values('Skill score').index[-1]
    df0 = df0.where(df0['Variable'] == var).dropna()
    if df is None:
        df = df0
    else:
        df = pd.concat([df, df0])
sns.barplot(data = df, x = "Variable", y = "Skill score", hue = "Variable", palette = "rocket")
plt.xticks(rotation = 90)
plt.title("Sequential feature selection")
plt.tight_layout()
plt.savefig("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/Diagnostics/SeqFeatSel.png")
with open("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/plots/R_VGAM/Diagnostics/SeqFeatSel.pkl", "wb") as f:
    pickle.dump(df, f)
plt.close()
