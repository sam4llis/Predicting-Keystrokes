import matplotlib as mpl
new_rc_params = {'text.usetex': True,
         'svg.fonttype': 'none',
         'text.latex.preamble': r'\usepackage{libertine}',
         'font.size': 7,
         'font.family': 'Linux Libertine',
         'mathtext.fontset': 'custom',
         'mathtext.rm': 'libertine',
         'mathtext.it': 'libertine:italic',
         'mathtext.bf': 'libertine:bold',
         'axes.linewidth': 0.1,
         'xtick.labelsize': 7,
         'ytick.labelsize': 7,
         'hatch.linewidth': 0.01,
         'legend.fontsize':7,
         'legend.handlelength': 2
         }
mpl.rcParams.update(new_rc_params)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import os
from utils import set_size

# GLOBAL
global width
width = 232.74377
width = 455.24411
save_dir = os.path.join('dissertation_article', 'res', 'img')
df_dir = os.path.join('code', 'src', 'data', 'dataframes')

transform_dict = {'VotingClassifier': 'ENS',
                  'RandomForestClassifier': 'RFC',
                  'SVC RBF kernel': 'SVC',
                  'GaussianNB': 'NBC'
                 }

def create_dataset_df(filepath_sm7b, filepath_xref, filepath_cpred):
    df_sm7b = pd.read_pickle(filepath_sm7b)
    df_sm7b['Dataset'] = df_sm7b.shape[0] * ['SM7B']
    df_xref = pd.read_pickle(filepath_xref)
    df_xref['Dataset'] = df_xref.shape[0] * ['XREF']
    df_cpred = pd.read_pickle(filepath_cpred)
    df_cpred['Dataset'] = df_cpred.shape[0] * ['Cross-Prediction']
    frames = [df_sm7b, df_xref, df_cpred]
    df = pd.concat(frames)
    df = pd.melt(df, id_vars=['Classifier', 'Dataset'], var_name='Metric')
    df['Classifier'].replace(transform_dict, inplace=True)
    return df

def create_user_df(user, *filepaths):
    df = create_dataset_df(*filepaths)
    df['User'] = [user] * df.shape[0]
    return df

def plot_metrics(df, x, y, hue, col, col_wrap=None, ylim=(0,1), aspect=1):
    g = sns.catplot(x=x, y=y, data=df, hue=hue, col=col, col_wrap=col_wrap,
            height=set_size(width)[1], aspect=aspect, legend_out=False, kind='bar',
            saturation=.5, sharex=False, ci=None)
    g._legend.remove()
    handles, labels = g.fig.get_axes()[1].get_legend_handles_labels()
    g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4,
            frameon=False)
    g.set_axis_labels('', 'Metric (\%)')
    g.set(ylim=ylim)
    g.despine(left=True)

    # Add annotations.
    for ax in g.axes.ravel():
        for c in ax.containers:
            labels = [f'{(v.get_height()):.2f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)



# --------------------------------------------------------------------------- #
# Plot Train Test Metrics vs. SMT Datasets
# ---------------------------------------------------------------------------- #

# User01.
fdir = os.path.join(df_dir, 'USER01', 'ECLF', 'TRAIN')
ECLF_SM7B_SMT_U1 = os.path.join(fdir, 'SMT_SM7B.pkl')
ECLF_XREF_SMT_U1 = os.path.join(fdir, 'SMT_XREF.pkl')
ECLF_CPRED_SMT_U1 = os.path.join(fdir, 'SMT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'TRAIN')
RFC_SM7B_SMT_U1 = os.path.join(fdir, 'SMT_SM7B.pkl')
RFC_XREF_SMT_U1 = os.path.join(fdir, 'SMT_XREF.pkl')
RFC_CPRED_SMT_U1 = os.path.join(fdir, 'SMT_CPRED.pkl')

# User02.
fdir = os.path.join(df_dir, 'USER02', 'ECLF', 'TRAIN')
ECLF_SM7B_SMT_U2 = os.path.join(fdir, 'SMT_SM7B.pkl')
ECLF_XREF_SMT_U2 = os.path.join(fdir, 'SMT_XREF.pkl')
ECLF_CPRED_SMT_U2 = os.path.join(fdir, 'SMT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'TRAIN')
RFC_SM7B_SMT_U2 = os.path.join(fdir, 'SMT_SM7B.pkl')
RFC_XREF_SMT_U2 = os.path.join(fdir, 'SMT_XREF.pkl')
RFC_CPRED_SMT_U2 = os.path.join(fdir, 'SMT_CPRED.pkl')

df1_eclf = create_user_df('USER01', ECLF_SM7B_SMT_U1, ECLF_XREF_SMT_U1,
        ECLF_CPRED_SMT_U1)
df1_rfc = create_user_df('USER01', RFC_SM7B_SMT_U1, RFC_XREF_SMT_U1,
        RFC_CPRED_SMT_U1)

df2_eclf = create_user_df('USER02', ECLF_SM7B_SMT_U2, ECLF_XREF_SMT_U2,
        ECLF_CPRED_SMT_U2)
df2_rfc = create_user_df('USER02', RFC_SM7B_SMT_U2, RFC_XREF_SMT_U2,
        RFC_CPRED_SMT_U2)

frames = [df1_eclf, df1_rfc, df2_eclf, df2_rfc]
df = pd.concat(frames)

plot_metrics(df=df, x='Classifier', y='value', hue='Metric', col='User',
        ylim=(0,.7))
# plt.show()

# --------------------------------------------------------------------------- #
# Plot Train Test Metrics vs. HPT Datasets
# ---------------------------------------------------------------------------- #

# User01.
fdir = os.path.join(df_dir, 'USER01', 'ECLF', 'TRAIN')
ECLF_SM7B_HP_U1 = os.path.join(fdir, 'HP_SM7B.pkl')
ECLF_XREF_HP_U1 = os.path.join(fdir, 'HP_XREF.pkl')
ECLF_CPRED_HP_U1 = os.path.join(fdir, 'HP_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'TRAIN')
RFC_SM7B_HP_U1 = os.path.join(fdir, 'HP_SM7B.pkl')
RFC_XREF_HP_U1 = os.path.join(fdir, 'HP_XREF.pkl')
RFC_CPRED_HP_U1 = os.path.join(fdir, 'HP_CPRED.pkl')

# User02.
fdir = os.path.join(df_dir, 'USER02', 'ECLF', 'TRAIN')
ECLF_SM7B_HP_U2 = os.path.join(fdir, 'HP_SM7B.pkl')
ECLF_XREF_HP_U2 = os.path.join(fdir, 'HP_XREF.pkl')
ECLF_CPRED_HP_U2 = os.path.join(fdir, 'HP_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'TRAIN')
RFC_SM7B_HP_U2 = os.path.join(fdir, 'HP_SM7B.pkl')
RFC_XREF_HP_U2 = os.path.join(fdir, 'HP_XREF.pkl')
RFC_CPRED_HP_U2 = os.path.join(fdir, 'HP_CPRED.pkl')

df1_eclf = create_user_df('USER01', ECLF_SM7B_HP_U1, ECLF_XREF_HP_U1,
        ECLF_CPRED_HP_U1)
df1_rfc = create_user_df('USER01', RFC_SM7B_HP_U1, RFC_XREF_HP_U1,
        RFC_CPRED_HP_U1)

df2_eclf = create_user_df('USER02', ECLF_SM7B_HP_U2, ECLF_XREF_HP_U2,
        ECLF_CPRED_HP_U2)
df2_rfc = create_user_df('USER02', RFC_SM7B_HP_U2, RFC_XREF_HP_U2,
        RFC_CPRED_HP_U2)

frames = [df1_eclf, df1_rfc, df2_eclf, df2_rfc]
df = pd.concat(frames)

plot_metrics(df=df, x='Classifier', y='value', hue='Metric', col='User',
        ylim=(0,.7))
# plt.show()

# ---------------------------------------------------------------------------- #
# Plot Train Test Metrics vs. TT Datasets
# ---------------------------------------------------------------------------- #

# User01
fdir = os.path.join(df_dir, 'USER01', 'ECLF', 'TRAIN')
ECLF_SM7B_TT_U1 = os.path.join(fdir, 'TT_SM7B.pkl')
ECLF_XREF_TT_U1 = os.path.join(fdir, 'TT_XREF.pkl')
ECLF_CPRED_TT_U1 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'TRAIN')
RFC_SM7B_TT_U1 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_TT_U1 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_TT_U1 = os.path.join(fdir, 'TT_CPRED.pkl')

# User02.
fdir = os.path.join(df_dir, 'USER02', 'ECLF', 'TRAIN')
ECLF_SM7B_TT_U2 = os.path.join(fdir, 'TT_SM7B.pkl')
ECLF_XREF_TT_U2 = os.path.join(fdir, 'TT_XREF.pkl')
ECLF_CPRED_TT_U2 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'TRAIN')
RFC_SM7B_TT_U2 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_TT_U2 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_TT_U2 = os.path.join(fdir, 'TT_CPRED.pkl')

df1_eclf = create_user_df('USER01', ECLF_SM7B_TT_U1, ECLF_XREF_TT_U1,
        ECLF_CPRED_TT_U1)
df1_rfc = create_user_df('USER01', RFC_SM7B_TT_U1, RFC_XREF_TT_U1,
        RFC_CPRED_TT_U1)

df2_eclf = create_user_df('USER02', ECLF_SM7B_TT_U2, ECLF_XREF_TT_U2,
        ECLF_CPRED_TT_U2)
df2_rfc = create_user_df('USER02', RFC_SM7B_TT_U2, RFC_XREF_TT_U2,
        RFC_CPRED_TT_U2)

frames = [df1_eclf, df1_rfc, df2_eclf, df2_rfc]
df = pd.concat(frames)

plot_metrics(df=df, x='Classifier', y='value', hue='Metric', col='User',
        ylim=(0,.7))
# # plt.show()

# ---------------------------------------------------------------------------- #
# Plot Train Test Metrics vs. Different Datasets
# ---------------------------------------------------------------------------- #
plt.close('all')
fdir = os.path.join(df_dir, 'TRAIN_TEST_METRICS')

filepath_sm7b = os.path.join(fdir, 'SM7B_METRICS.pkl')
filepath_xref = os.path.join(fdir, 'XREF_METRICS.pkl')
filepath_cpred = os.path.join(fdir,'CPRED_METRICS.pkl')

df = create_dataset_df(filepath_sm7b, filepath_xref, filepath_cpred)

width = 455.24411/2
plot_metrics(df=df, x='Dataset', y='value', hue='Metric', col='Classifier',
        col_wrap=2, aspect=2)

fpath = os.path.join(save_dir, 'train_test_classifier_metrics')
plt.savefig(fpath, bbox_inches='tight', dpi=1200)
plt.show()
width = 455.24411

# ---------------------------------------------------------------------------- #
# (ALTERNATIVE) PLOT CLASSIFIER vs. USER DATASETS FOR EACH USER
# ---------------------------------------------------------------------------- #
plt.close('all')

# User01
fdir = os.path.join(df_dir, 'USER01', 'RFC', 'TRAIN')
RFC_SM7B_SMT_U1 = os.path.join(fdir, 'SMT_SM7B.pkl')
RFC_XREF_SMT_U1 = os.path.join(fdir, 'SMT_XREF.pkl')
RFC_CPRED_SMT_U1 = os.path.join(fdir, 'SMT_CPRED.pkl')

RFC_SM7B_HP_U1 = os.path.join(fdir, 'HP_SM7B.pkl')
RFC_XREF_HP_U1 = os.path.join(fdir, 'HP_XREF.pkl')
RFC_CPRED_HP_U1 = os.path.join(fdir, 'HP_CPRED.pkl')

RFC_SM7B_TT_U1 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_TT_U1 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_TT_U1 = os.path.join(fdir, 'TT_CPRED.pkl')

# User02
fdir = os.path.join(df_dir, 'USER02', 'RFC', 'TRAIN')
RFC_SM7B_SMT_U2 = os.path.join(fdir, 'SMT_SM7B.pkl')
RFC_XREF_SMT_U2 = os.path.join(fdir, 'SMT_XREF.pkl')
RFC_CPRED_SMT_U2 = os.path.join(fdir, 'SMT_CPRED.pkl')

RFC_SM7B_HP_U2 = os.path.join(fdir, 'HP_SM7B.pkl')
RFC_XREF_HP_U2 = os.path.join(fdir, 'HP_XREF.pkl')
RFC_CPRED_HP_U2 = os.path.join(fdir, 'HP_CPRED.pkl')

RFC_SM7B_TT_U2 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_TT_U2 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_TT_U2 = os.path.join(fdir, 'TT_CPRED.pkl')

df1_smt_rfc = create_user_df('01',
        RFC_SM7B_SMT_U1, RFC_XREF_SMT_U1, RFC_CPRED_SMT_U1)
df1_smt_rfc['Typing Style'] = df1_smt_rfc.shape[0] * ['SMT']

df1_hp_rfc = create_user_df('01',
        RFC_SM7B_HP_U1, RFC_XREF_HP_U1, RFC_CPRED_HP_U1)
df1_hp_rfc['Typing Style'] = df1_smt_rfc.shape[0] * ['HP']

df1_tt_rfc = create_user_df('01',
        RFC_SM7B_TT_U1, RFC_XREF_TT_U1, RFC_CPRED_TT_U1)
df1_tt_rfc['Typing Style'] = df1_smt_rfc.shape[0] * ['TT']

df2_smt_rfc = create_user_df('02',
        RFC_SM7B_SMT_U2, RFC_XREF_SMT_U2, RFC_CPRED_SMT_U2)
df2_smt_rfc['Typing Style'] = df2_smt_rfc.shape[0] * ['SMT']

df2_hp_rfc = create_user_df('02',
        RFC_SM7B_HP_U2, RFC_XREF_HP_U2, RFC_CPRED_HP_U2)
df2_hp_rfc['Typing Style'] = df2_smt_rfc.shape[0] * ['HP']

df2_tt_rfc = create_user_df('02',
        RFC_SM7B_TT_U2, RFC_XREF_TT_U2, RFC_CPRED_TT_U2)
df2_tt_rfc['Typing Style'] = df2_smt_rfc.shape[0] * ['TT']

frames = [df1_smt_rfc, df1_hp_rfc, df1_tt_rfc, df2_smt_rfc, df2_hp_rfc, df2_tt_rfc]
df = pd.concat(frames)
df = df.loc[df['Dataset']=='Cross-Prediction']

plot_metrics(df=df, x='Typing Style', y='value', hue='Metric', col='User',
        ylim=(0,.7))

fpath = os.path.join(save_dir, 'user_dataset_classifier_metrics')
plt.savefig(fpath, bbox_inches='tight', dpi=1200)
plt.show()

# ---------------------------------------------------------------------------- #
# (ALTERNATIVE) PLOT TT vs. DIFFERENT TRAIN DATASETS FOR EACH USER
# ---------------------------------------------------------------------------- #
plt.close('all')

# User01
fdir = os.path.join(df_dir, 'USER01', 'RFC', 'SMT')
RFC_SM7B_SMT_U1 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_SMT_U1 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_SMT_U1 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'HP')
RFC_SM7B_HP_U1 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_HP_U1 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_HP_U1 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'SMT+HP')
RFC_SM7B_SMTHP_U1 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_SMTHP_U1 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_SMTHP_U1 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'SMT+HP+TRAIN')
RFC_SM7B_SMTHPTRAIN_U1 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_SMTHPTRAIN_U1 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_SMTHPTRAIN_U1 = os.path.join(fdir, 'TT_CPRED.pkl')

# User02
fdir = os.path.join(df_dir, 'USER02', 'RFC', 'SMT')
RFC_SM7B_SMT_U2 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_SMT_U2 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_SMT_U2 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'HP')
RFC_SM7B_HP_U2 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_HP_U2 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_HP_U2 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'SMT+HP')
RFC_SM7B_SMTHP_U2 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_SMTHP_U2 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_SMTHP_U2 = os.path.join(fdir, 'TT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'SMT+HP+TRAIN')
RFC_SM7B_SMTHPTRAIN_U2 = os.path.join(fdir, 'TT_SM7B.pkl')
RFC_XREF_SMTHPTRAIN_U2 = os.path.join(fdir, 'TT_XREF.pkl')
RFC_CPRED_SMTHPTRAIN_U2 = os.path.join(fdir, 'TT_CPRED.pkl')


df1_smt_rfc = create_user_df('01',
        RFC_SM7B_SMT_U1, RFC_XREF_SMT_U1, RFC_CPRED_SMT_U1)
df1_smt_rfc['Training Style'] = df1_smt_rfc.shape[0] * ['SMT']

df1_hp_rfc = create_user_df('01',
        RFC_SM7B_HP_U1, RFC_XREF_HP_U1, RFC_CPRED_HP_U1)
df1_hp_rfc['Training Style'] = df1_smt_rfc.shape[0] * ['HP']

df1_smthp_rfc = create_user_df('01',
        RFC_SM7B_SMTHP_U1, RFC_XREF_SMTHP_U1, RFC_CPRED_SMTHP_U1)
df1_smthp_rfc['Training Style'] = df1_smt_rfc.shape[0] * ['SMT \& HP']

df1_smthptrain_rfc = create_user_df('01',
        RFC_SM7B_SMTHPTRAIN_U1, RFC_XREF_SMTHPTRAIN_U1, RFC_CPRED_SMTHPTRAIN_U1)
df1_smthptrain_rfc['Training Style'] = df1_smt_rfc.shape[0] * ['SMT \& HP \& TRAINING']


df2_smt_rfc = create_user_df('02',
        RFC_SM7B_SMT_U2, RFC_XREF_SMT_U2, RFC_CPRED_SMT_U2)
df2_smt_rfc['Training Style'] = df2_smt_rfc.shape[0] * ['SMT']

df2_hp_rfc = create_user_df('02',
        RFC_SM7B_HP_U2, RFC_XREF_HP_U2, RFC_CPRED_HP_U2)
df2_hp_rfc['Training Style'] = df2_smt_rfc.shape[0] * ['HP']

df2_smthp_rfc = create_user_df('02',
        RFC_SM7B_SMTHP_U2, RFC_XREF_SMTHP_U2, RFC_CPRED_SMTHP_U2)
df2_smthp_rfc['Training Style'] = df2_smt_rfc.shape[0] * ['SMT \& HP']

df2_smthptrain_rfc = create_user_df('02',
        RFC_SM7B_SMTHPTRAIN_U2, RFC_XREF_SMTHPTRAIN_U2, RFC_CPRED_SMTHPTRAIN_U2)
df2_smthptrain_rfc['Training Style'] = df2_smt_rfc.shape[0] * ['SMT \& HP \& TRAINING']

frames = [df1_smt_rfc, df1_hp_rfc, df1_smthp_rfc, df1_smthptrain_rfc,
        df2_smt_rfc, df2_hp_rfc, df2_smthp_rfc, df2_smthptrain_rfc]
df = pd.concat(frames)
df = df.loc[df['Dataset']=='Cross-Prediction']

plot_metrics(df=df, x='Training Style', y='value', hue='Metric', col='User',
        ylim=(0,.7))
fpath=os.path.join(save_dir, 'user_tt_metrics')
plt.savefig(fpath, dpi=1200, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------- #
# (ALTERNATIVE) PLOT TT vs. DIFFERENT TRAIN DATASETS FOR EACH USER
# ---------------------------------------------------------------------------- #
plt.close('all')

# User01
fdir = os.path.join(df_dir, 'USER01', 'RFC', 'SMT_48')
RFC_SM7B_SMT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_SMT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_SMT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'HP_48')
RFC_SM7B_HP_U1 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_HP_U1 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_HP_U1 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'TT_48')
RFC_SM7B_TT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_TT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_TT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

# fdir = os.path.join(df_dir, 'USER01', 'RFC', 'SMT+HP_48')
# RFC_SM7B_SMTHP_U1 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
# RFC_XREF_SMTHP_U1 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
# RFC_CPRED_SMTHP_U1 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

# fdir = os.path.join(df_dir, 'USER01', 'RFC', 'TRAIN')
# RFC_SM7B_TRAIN_U1 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
# RFC_XREF_TRAIN_U1 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
# RFC_CPRED_TRAIN_U1 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

# fdir = os.path.join(df_dir, 'USER01', 'RFC', 'HP+TT+TRAIN')
# RFC_SM7B_HPTTTRAIN_U1 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
# RFC_XREF_HPTTTRAIN_U1 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
# RFC_CPRED_HPTTTRAIN_U1 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER01', 'RFC', 'HP+TT_48')
RFC_SM7B_HPTT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_HPTT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_HPTT_U1 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

# User02
fdir = os.path.join(df_dir, 'USER02', 'RFC', 'SMT_48')
RFC_SM7B_SMT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_SMT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_SMT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'HP_48')
RFC_SM7B_HP_U2 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_HP_U2 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_HP_U2 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'TT_48')
RFC_SM7B_TT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_TT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_TT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

# fdir = os.path.join(df_dir, 'USER02', 'RFC', 'SMT+HP_48')
# RFC_SM7B_SMTHP_U2 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
# RFC_XREF_SMTHP_U2 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
# RFC_CPRED_SMTHP_U2 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

# fdir = os.path.join(df_dir, 'USER02', 'RFC', 'TRAIN')
# RFC_SM7B_TRAIN_U2 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
# RFC_XREF_TRAIN_U2 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
# RFC_CPRED_TRAIN_U2 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

# fdir = os.path.join(df_dir, 'USER02', 'RFC', 'HP+TT+TRAIN')
# RFC_SM7B_HPTTTRAIN_U2 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
# RFC_XREF_HPTTTRAIN_U2 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
# RFC_CPRED_HPTTTRAIN_U2 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

fdir = os.path.join(df_dir, 'USER02', 'RFC', 'HP+TT_48')
RFC_SM7B_HPTT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_SM7B.pkl')
RFC_XREF_HPTT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_XREF.pkl')
RFC_CPRED_HPTT_U2 = os.path.join(fdir, 'SAMPLE_TEXT_CPRED.pkl')

df1_SMT_rfc = create_user_df('01', RFC_SM7B_SMT_U1, RFC_XREF_SMT_U1, RFC_CPRED_SMT_U1)
df1_SMT_rfc['Training Style'] = df1_smt_rfc.shape[0] * ['SMT']

df1_HP_rfc = create_user_df('01', RFC_SM7B_HP_U1, RFC_XREF_HP_U1, RFC_CPRED_HP_U1)
df1_HP_rfc['Training Style'] = df1_HP_rfc.shape[0] * ['HP']

df1_TT_rfc = create_user_df('01', RFC_SM7B_TT_U1, RFC_XREF_TT_U1, RFC_CPRED_TT_U1)
df1_TT_rfc['Training Style'] = df1_TT_rfc.shape[0] * ['TT']

# df1_SMTHP_rfc = create_user_df('01', RFC_SM7B_SMTHP_U1, RFC_XREF_SMTHP_U1, RFC_CPRED_SMTHP_U1)
# df1_SMTHP_rfc['Training Style'] = df1_SMTHP_rfc.shape[0] * ['SMT+HP']

# df1_TRAIN_rfc = create_user_df('01', RFC_SM7B_TRAIN_U1, RFC_XREF_TRAIN_U1, RFC_CPRED_TRAIN_U1)
# df1_TRAIN_rfc['Training Style'] = df1_TRAIN_rfc.shape[0] * ['TRAIN']

# df1_HPTTTRAIN_rfc = create_user_df('01', RFC_SM7B_HPTTTRAIN_U1, RFC_XREF_HPTTTRAIN_U1, RFC_CPRED_HPTTTRAIN_U1)
# df1_HPTTTRAIN_rfc['Training Style'] = df1_HPTTTRAIN_rfc.shape[0] * ['HP+TT+TRAIN']

df1_HPTT_rfc = create_user_df('01', RFC_SM7B_HPTT_U1, RFC_XREF_HPTT_U1, RFC_CPRED_HPTT_U1)
df1_HPTT_rfc['Training Style'] = df1_HPTT_rfc.shape[0] * ['HP+TT']


df2_SMT_rfc = create_user_df('02', RFC_SM7B_SMT_U2, RFC_XREF_SMT_U2, RFC_CPRED_SMT_U2)
df2_SMT_rfc['Training Style'] = df2_smt_rfc.shape[0] * ['SMT']

df2_HP_rfc = create_user_df('02', RFC_SM7B_HP_U2, RFC_XREF_HP_U2, RFC_CPRED_HP_U2)
df2_HP_rfc['Training Style'] = df2_HP_rfc.shape[0] * ['HP']

df2_TT_rfc = create_user_df('02', RFC_SM7B_TT_U2, RFC_XREF_TT_U2, RFC_CPRED_TT_U2)
df2_TT_rfc['Training Style'] = df2_TT_rfc.shape[0] * ['TT']

# df2_SMTHP_rfc = create_user_df('02', RFC_SM7B_SMTHP_U2, RFC_XREF_SMTHP_U2, RFC_CPRED_SMTHP_U2)
# df2_SMTHP_rfc['Training Style'] = df2_SMTHP_rfc.shape[0] * ['SMT+HP']

# df2_TRAIN_rfc = create_user_df('02', RFC_SM7B_TRAIN_U2, RFC_XREF_TRAIN_U2, RFC_CPRED_TRAIN_U2)
# df2_TRAIN_rfc['Training Style'] = df2_TRAIN_rfc.shape[0] * ['TRAIN']

# df2_HPTTTRAIN_rfc = create_user_df('02', RFC_SM7B_HPTTTRAIN_U2, RFC_XREF_HPTTTRAIN_U2, RFC_CPRED_HPTTTRAIN_U2)
# df2_HPTTTRAIN_rfc['Training Style'] = df2_HPTTTRAIN_rfc.shape[0] * ['HP+TT+TRAIN']

df2_HPTT_rfc = create_user_df('02', RFC_SM7B_HPTT_U2, RFC_XREF_HPTT_U2, RFC_CPRED_HPTT_U2)
df2_HPTT_rfc['Training Style'] = df2_HPTT_rfc.shape[0] * ['HP+TT']


frames = [df1_SMT_rfc, df1_HP_rfc, df1_TT_rfc,
          df1_HPTT_rfc,
          df2_SMT_rfc, df2_HP_rfc, df2_TT_rfc,
          df2_HPTT_rfc]

df = pd.concat(frames)
df = df.loc[df['Metric']=='Accuracy']
print(df.to_string())
g = plot_metrics(df=df, x='Training Style', y='value', hue='Dataset', col='User',
        ylim=(0, 1))
# g.set_axis_labels('', 'Accuracy (\%)')
fpath = os.path.join(save_dir, 'sample_text_classifier')
plt.savefig(fpath, bbox_inches='tight', dpi=1200)

plt.show()
