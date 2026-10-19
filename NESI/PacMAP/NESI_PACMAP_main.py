from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from IPython.display import display

SCRIPT_DIR = Path("/home/ayush/Desktop/NESI_review_material/PacMAP")
IN_CSV = SCRIPT_DIR / 'NESI_pacmap_coords.csv'
ROTATION_DEGREES = 90
FLIP_VERTICAL = True
JITTER_COPIES = 4
JITTER_SIGMA_FRAC = 0.006
DOT_SIZE = 4.0
BURST_THRESHOLD = 0.5
IIIC_PALETTE = {'Seizure':'#E13238','LPD':'#F08C2A','GPD':'#F2D549','LRDA':'#AABF45','GRDA':'#7AC8E3','Other':'#5C3A87','Burst':'#000000'}
IIIC_LEGEND_ORDER = ['Burst','Seizure','LPD','GPD','LRDA','GRDA','Other']
CAT_PROB_COL = {'Burst':'Burst_vs_NoBurst','Seizure':'IIIC_Seizure','LPD':'IIIC_LPD','GPD':'IIIC_GPD','LRDA':'IIIC_LRDA','GRDA':'IIIC_GRDA','Other':'Awake'}

def category_cmap(cat):
    base = IIIC_PALETTE[cat]
    pale = tuple(0.8 + 0.2*np.array(to_rgb(base)))
    if cat == 'Other':
        pale = tuple(0.85 + 0.15*np.array(to_rgb(base)))
        return LinearSegmentedColormap.from_list(cat,[base,pale])
    return LinearSegmentedColormap.from_list(cat,[pale,base])

def make_labels_rgb(df):
    cols = ['IIIC_Seizure','IIIC_LPD','IIIC_GPD','IIIC_LRDA','IIIC_GRDA','IIIC_Other']
    names = np.array(['Seizure','LPD','GPD','LRDA','GRDA','Other'])
    labels = names[np.argmax(df[cols].to_numpy(),axis=1)].copy()
    labels[df['Burst_vs_NoBurst'].to_numpy() >= BURST_THRESHOLD] = 'Burst'
    rgb = np.zeros((len(df),3))
    for cat in IIIC_PALETTE:
        mask = labels == cat
        if not mask.any(): continue
        probs = df.loc[mask,CAT_PROB_COL[cat]].to_numpy()
        lo,hi = np.quantile(probs,[0.05,0.95]) if len(probs)>20 else (probs.min(),probs.max())
        if hi-lo<1e-6: hi=lo+1e-6
        rgb[mask] = category_cmap(cat)(np.clip((probs-lo)/(hi-lo),0,1))[:,:3]
    return labels,rgb

def main():
    df = pd.read_csv(IN_CSV)
    xy = df[['pacmap_x_nesi','pacmap_y_nesi']].to_numpy(dtype=float)
    theta = np.deg2rad(ROTATION_DEGREES)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xy_rot = (xy-xy.mean(axis=0)) @ R.T

    # Flip vertically after rotation (preserves left-right orientation)
    if FLIP_VERTICAL:
        xy_rot[:,1] *= -1

    labels,rgb = make_labels_rgb(df)
    rng = np.random.default_rng(0)
    sx,sy = xy_rot.std(axis=0)*JITTER_SIGMA_FRAC
    xy_aug = np.vstack([xy_rot]+[xy_rot+np.column_stack([rng.normal(0,sx,len(df)),rng.normal(0,sy,len(df))]) for _ in range(JITTER_COPIES)])
    rgb_aug = np.tile(rgb,(JITTER_COPIES+1,1))
    labels_aug = np.tile(labels,JITTER_COPIES+1)
    nesi_aug = np.tile(df['NESI'].to_numpy(),JITTER_COPIES+1)
    counts = pd.Series(labels).value_counts()

    fig = plt.figure(figsize=(14,6.5),dpi=140)
    gs = fig.add_gridspec(1,2,wspace=0.06,left=0.03,right=0.97,top=0.94,bottom=0.04)
    axA,axB = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])

    order = np.argsort(np.abs(nesi_aug))
    axA.scatter(xy_aug[order,0],xy_aug[order,1],c=nesi_aug[order],cmap='RdBu_r',vmin=-3,vmax=3,s=DOT_SIZE,marker='o',edgecolors='none',alpha=0.55)

    for cat in ['Other','GRDA','LRDA','GPD','LPD','Seizure','Burst']:
        mask = labels_aug == cat
        axB.scatter(xy_aug[mask,0],xy_aug[mask,1],c=rgb_aug[mask],s=DOT_SIZE,marker='o',edgecolors='none',alpha=0.55)

    handles = [mpatches.Patch(color=IIIC_PALETTE[cat],label=f'{cat} (n={int(counts.get(cat,0))})') for cat in IIIC_LEGEND_ORDER]
    axB.legend(handles=handles,loc='upper right',fontsize=7,frameon=False,title='IIIC category',title_fontsize=8)

    for ax,title in [(axA,'A. NESI (continuous)'),(axB,'B. Dominant IIIC category')]:
        ax.set_title(title,fontsize=12,fontweight='bold')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_aspect('equal',adjustable='box')
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.savefig(SCRIPT_DIR / "NESI_pacmap.png", dpi=500, bbox_inches='tight')

    display(fig)
    plt.close(fig)
    print(f'Input: {len(df):,} windows | Rotation: {ROTATION_DEGREES}° CCW | Vertical flip: {FLIP_VERTICAL}')

if __name__ == '__main__':
    main()