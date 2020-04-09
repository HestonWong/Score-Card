from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, fbeta_score
from toad.selection import select, stepwise
from toad.transform import Combiner, WOETransformer
from toad.plot import bin_plot
from toad.metrics import KS_bucket, PSI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import visuals as vs
import time, toad

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
start = time.time()

Q1 = pd.read_csv('D:/upgrade/项目/风控/D_银行贷款/data/LoanStats_2017Q1.csv', header=1, low_memory=False)[:-2]
Q2 = pd.read_csv('D:/upgrade/项目/风控/D_银行贷款/data/LoanStats_2017Q2.csv', header=1, low_memory=False)[:-2]
Q3 = pd.read_csv('D:/upgrade/项目/风控/D_银行贷款/data/LoanStats_2017Q3.csv', header=1, low_memory=False)[:-2]
Q4 = pd.read_csv('D:/upgrade/项目/风控/D_银行贷款/data/LoanStats_2017Q4.csv', header=1, low_memory=False)[:-2]
Q1['split'] = 'Q1'
Q2['split'] = 'Q2'
Q3['split'] = 'Q3'
Q4['split'] = 'Q4'
data = pd.concat([Q1, Q2, Q3, Q4], join='inner')
d = ['desc', 'emp_title', 'tax_liens', 'last_pymnt_d', 'last_credit_pull_d', 'issue_d', 'zip_code', 'grade', 'debt_settlement_flag', 'title',
     'debt_settlement_flag_date', 'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 'last_pymnt_amnt', 'total_pymnt', 'total_pymnt_inv',
     'total_rec_int', 'total_rec_late_fee', 'int_rate', 'term', 'next_pymnt_d', 'il_util', 'pymnt_plan', 'initial_list_status', 'application_type',
     'out_prncp', 'out_prncp_inv', 'funded_amnt', 'funded_amnt_inv', 'earliest_cr_line', 'num_tl_120dpd_2m', 'hardship_amount', 'hardship_dpd',
     'hardship_end_date', 'hardship_flag', 'hardship_last_payment_amount', 'hardship_length', 'hardship_loan_status', 'url', 'id', 'installment',
     'hardship_payoff_balance_amount', 'hardship_reason', 'hardship_start_date', 'hardship_status', 'hardship_type', 'total_bal_ex_mort',
     'loan_amnt']  # ['last_fico_range_high','fico_range_high','last_fico_range_low']
data = data.drop(d, axis=1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 清洗
print('数据清洗处理中'.center(60, '—'))
# 字符型清洗
data[['revol_util']] = data[['revol_util']].apply(lambda y: y.str.rstrip('%').astype(float) / 100)

# 删除只有唯一值的特征
only_one = data.columns[data.nunique() == 1].tolist()
data.drop(only_one, axis=1, inplace=True)
# print(data.columns[data.nunique() == 1].tolist())

# 删除灰色区域,加大区分度
data['loan_status'] = data['loan_status'].map({
    'Fully Paid': 0,
    'Current': 0,
    'Charged Off': 1,
    'Late (31-120 days)': 1,
    'Late (16-30 days)': 1,
    'In Grace Period': 2,
    'Default': 2})
data = data[data['loan_status'].isin([0, 1])]
print('处理完成，数据共有{}行，{}列'.format(data.shape[0], data.shape[1]), '\n' * 2)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 缺失
print('缺失值处理中'.center(60, '—'))

before = data['loan_status'].groupby(data['loan_status']).count()
all = data.shape[0]
a1, b1 = before[0], before[1]
data = data[data.apply(lambda x: x.isnull().sum() < 31, axis=1)]  # 删除个案缺失
after = data['loan_status'].groupby(data['loan_status']).count()
a2, b2 = after[0], after[1]
print(before)
print('正常用户删除：{}，占比{:.2f}% \n坏用户删除：{}，占比{:.2f}%'.format((a1 - a2), ((a1 - a2) * 100 / all), (b1 - b2), ((b1 - b2) * 100 / all)))

missing_80percent = list(data.columns[data.isnull().sum() > len(data) * 0.7])
fill_min = ['mo_sin_old_il_acct', 'bc_util', 'bc_open_to_buy', 'mths_since_recent_bc']
fill_max = ['mths_since_rcnt_il', 'percent_bc_gt_75', 'mths_since_recent_inq', 'revol_util', 'all_util', 'avg_cur_bal',
            'mths_since_recent_revol_delinq']
fill_median = ['mths_since_last_delinq', 'dti']

data.drop(missing_80percent, axis=1, inplace=True)
data['emp_length'].fillna('< 1 year', inplace=True)
data[fill_min] = data[fill_min].fillna(data[fill_min].max())
data[fill_max] = data[fill_max].fillna(data[fill_max].max())
data[fill_median] = data[fill_median].fillna(data[fill_median].max())

# missing1 = data.columns[data.isnull().sum() != 0].tolist()
# missing2 = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
# print(missing1, '\n', missing2)
# data.to_csv('data.csv', index=False)

print('处理完成，数据共有{}行，{}列'.format(data.shape[0], data.shape[1]), '\n' * 2)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 特征筛选A
print('特征第一次筛选'.center(60, '—'))
train = data[data['split'].isin(['Q1', 'Q2', 'Q3'])].drop('split', axis=1)
test = data[data['split'].isin(['Q4'])].drop('split', axis=1)

train_s, drops = select(train, target='loan_status', iv=0.005, corr=0.8, return_drop=True)
test_s = test[train_s.columns]
print('IV筛选不通过的特征为：\n', drops['iv'], '\n',
      'corr筛选不通过的特征为：\n', drops['corr'])
print('处理完成，剩余{}特征'.format(train_s.shape[1]), '\n' * 2)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 分箱
print('卡方分箱中'.center(60, '—'))
comb = Combiner()
columns = train_s.columns


def combine(data, target, columns=[], exclude=[]):  # 精细化分箱
    for i in columns[~columns.isin(exclude)]:
        data_i = pd.concat([data[i], data[target]], axis=1)
        comb.fit(data_i, y=target, method='chi', min_samples=0.1)
        bins = comb.export()
        print(bins)
        data_c = comb.transform(data_i, labels=True)
        bin_plot(data_c, x=i, target=target)
        plt.show()


# combine(train_s, target='loan_status', columns=columns, exclude=['loan_status'])

comb.fit(train_s, y='loan_status', method='chi', min_samples=0.1)
rules = {
    'emp_length': [['< 1 year'], ['1 year', '2 years', '3 years'], ['4 years', '5 years', '6 years', '7 years', '8 years'], ['9 years', '10+ years']],
    'percent_bc_gt_75': [11.1, 25.9, 52.0],
    'avg_cur_bal': [6515.0, 10622.0, 19486.0, 36453.0]}
comb.set_rules(rules)
train_b = comb.transform(train_s, labels=True)
test_b = comb.transform(test_s, labels=True)
for i in columns[~columns.isin(['split', 'loan_status'])]:
    data_i = pd.concat([train_b[i], train_s['loan_status']], axis=1)
    bin_plot(data_i, x=i, target='loan_status')
    plt.show()
print('分箱完成', '\n' * 2)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# WOE编码
print('WOE编码'.center(62, '—'))
WOE = WOETransformer()
train_w = WOE.fit_transform(train_b, y='loan_status')
test_w = WOE.fit_transform(test_b, y='loan_status')
print('WOE编码完成', '\n' * 2)
train_w.to_csv('train_w.csv', index=False)
test_w.to_csv('test_w.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 特征筛选B
print('特征第二次筛选'.center(60, '—'))
train_w = pd.read_csv('train_w.csv')
test_w = pd.read_csv('test_w.csv')

train_s2, drops = select(train_w, target='loan_status', iv=0.005, corr=0.8, return_drop=True)
test_s2 = test_w[train_s2.columns]
print('IV筛选不通过的特征为：\n', drops['iv'], '\n',
      'corr筛选不通过的特征为：\n', drops['corr'])
print('处理完成，剩余{}特征'.format(train_s2.shape[1]))

print('Logistic逐步回归筛选中')
train_step = stepwise(train_s2, target='loan_status', estimator='ols', direction='both', criterion='aic')
test_step = test_s2[train_step.columns]
print('处理完成，剩余{}特征'.format(train_step.shape[1]), '\n' * 2)
# data_step = pd.concat([train_step, test_step], join='inner')
# data_step.to_csv('data_step.csv', index=False)
train_step.to_csv('train_step.csv', index=False)
test_step.to_csv('test_step.csv', index=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 模型训练
print('模型训练'.center(60, '—'))
train_step = pd.read_csv('train_step.csv')
test_step = pd.read_csv('test_step.csv')
print(train_step['loan_status'].groupby(train_step['loan_status']).count())
print('总体违约率：{:.1f}%'.format(sum(train_step['loan_status'] == 1) * 100 / train_step.shape[0]), '\n')
features = train_step.columns.tolist()
features.remove('loan_status')
X_train, X_test, y_train, y_test = train_step[features], test_step[features], train_step['loan_status'], test_step['loan_status']

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
class_weight = [{0: 1, 1: 27.1319}, {0: 1, 1: 28}, {0: 1, 1: 30}, {0: 1, 1: 32}, {0: 1, 1: 34}, {0: 1, 1: 36}, {0: 1, 1: 38}, {0: 1, 1: 40},
                {0: 1, 1: 42}, {0: 1, 1: 44}, {0: 1, 1: 46}, {0: 1, 1: 48}, {0: 1, 1: 50}, {0: 1, 1: 52}, {0: 1, 1: 54}]
AUC_KS_Recall_df = pd.DataFrame(
    columns=['index', 'AUC_train', 'AUC_test', 'KS_train', 'KS_test', 'Fbeta_train', 'Fbeta_test', 'precision_train', 'precision_test',
             'Recall_train', 'Recall_test'])
n = 0
# for i in scoring:
#     for j in class_weight:
#         print('Scoring:{},Weight:{}'.format(i, j))
#         lr = LogisticRegressionCV(Cs=10, scoring=i, class_weight=j, max_iter=3000, cv=4, n_jobs=-1, random_state=47)
#         lr.fit(X_train, y_train)
#
#         y_train_pred = lr.predict(X_train)
#         y_train_prob = lr.predict_proba(X_train)[:, 1]
#         y_test_pred = lr.predict(X_test)
#         y_test_prob = lr.predict_proba(X_test)[:, 1]
#         fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
#         fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
#
#         AUC_train, AUC_test = auc(fpr_train, tpr_train), auc(fpr_test, tpr_test)
#         KS_train, KS_test = max(abs(tpr_train - fpr_train)), max(abs(tpr_test - fpr_test))
#         Fbeta_train, Fbeta_test = fbeta_score(y_train, y_train_pred, 2), fbeta_score(y_test, y_test_pred, 2)
#         precision_train, precision_test = precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)
#         Recall_train, Recall_test = recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)
#
#         AUC_KS_Recall_df.loc[n, 'index'] = 'S:' + str(i) + ';' + 'W:' + str(j[0]) + '/' + str(j[1]) + ';' + 'C:' + str(np.round(lr.C_[0], 5))
#         AUC_KS_Recall_df.loc[n, 'AUC_train'], AUC_KS_Recall_df.loc[n, 'AUC_test'] = AUC_train, AUC_test
#         AUC_KS_Recall_df.loc[n, 'KS_train'], AUC_KS_Recall_df.loc[n, 'KS_test'] = KS_train, KS_test
#         AUC_KS_Recall_df.loc[n, 'Fbeta_train'], AUC_KS_Recall_df.loc[n, 'Fbeta_test'] = Fbeta_train, Fbeta_test
#         AUC_KS_Recall_df.loc[n, 'precision_train'], AUC_KS_Recall_df.loc[n, 'precision_test'] = precision_train, precision_test
#         AUC_KS_Recall_df.loc[n, 'Recall_train'], AUC_KS_Recall_df.loc[n, 'Recall_test'] = Recall_train, Recall_test
#         n += 1
#
#         print('模型训练完成：')
#         print('最佳C值:{}'.format(lr.C_))
#         print('AUC_train:{:.3f} | AUC_test:{:.3f}'.format(AUC_train, AUC_test))
#         print('KS_train:{:.3f} | KS_test:{:.3f}'.format(KS_train, KS_test))
#         print('Fbeta_train:{:.3f} | Fbeta_test:{:.3f}'.format(Fbeta_train, Fbeta_test))
#         print('precision_train:{:.3f} | precision_test:{:.3f}'.format(precision_train, precision_test))
#         print('Recall_train:{:.3f} | Recall_test:{:.3f}'.format(Recall_train, Recall_test), '\n' * 2)

# AUC_KS_Recall_df.to_csv('AUC_KS_Recall_df.csv')

lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, n_jobs=-1, random_state=47)
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_train_prob = lr.predict_proba(X_train)[:, 1]
y_test_pred = lr.predict(X_test)
y_test_prob = lr.predict_proba(X_test)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

AUC_train, AUC_test = auc(fpr_train, tpr_train), auc(fpr_test, tpr_test)
KS_train, KS_test = max(abs(tpr_train - fpr_train)), max(abs(tpr_test - fpr_test))
Fbeta_train, Fbeta_test = fbeta_score(y_train, y_train_pred, 2), fbeta_score(y_test, y_test_pred, 2)
precision_train, precision_test = precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)
Recall_train, Recall_test = recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)

print('模型训练完成：')
print('AUC_train:{:.3f} | AUC_test:{:.3f}'.format(AUC_train, AUC_test))
print('KS_train:{:.3f} | KS_test:{:.3f}'.format(KS_train, KS_test))
print('Fbeta_train:{:.3f} | Fbeta_test:{:.3f}'.format(Fbeta_train, Fbeta_test))
print('precision_train:{:.3f} | precision_test:{:.3f}'.format(precision_train, precision_test))
print('Recall_train:{:.3f} | Recall_test:{:.3f}'.format(Recall_train, Recall_test), '\n' * 2)
vs.roc(fpr_train, tpr_train, fpr_test, tpr_test)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 报告
print('报告'.center(62, '—'))

# EDA报告，参与模型训练的变量情况
EDA = toad.detect(X_train)
EDA.to_csv('EDA.csv')
print('EDA done!')

# PSI
PSI = PSI(X_train, X_test)
PSI.to_csv('PSI.csv')
print('PSI done!')

# KS报告
KS_train = KS_bucket(y_train_prob, y_train, bucket=20, method='quantile')
KS_test = KS_bucket(y_test_prob, y_test, bucket=20, method='quantile')
KS_train.to_csv('KS_train.csv')
KS_test.to_csv('KS_test.csv')
print('KS done!')

end = time.time()
print('总耗时：{:.2f}'.format((end - start)))
