import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt

MECHINE_NAME = "frontier"
# MECHINE_NAME = "pm"

def mean_absolute_percentage_error(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true != 0)
    if not np.any(mask):
        return None
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def direction_accuracy_threshold(y_true, y_pred, threshold_percent=0.02):
    """
    做法 B：只对幅度足够大的真实值变化(> threshold_percent)计算方向准确度(DA)。
    threshold_percent=0.02 表示相邻点变化幅度若不超过2% (基于前一时刻的真实值)，则跳过。
    
    返回值为在这些"显著变化"点上的方向预测准确度 (0~1)，
    如果整个序列都没有超过阈值的变化，则返回 None 以示无法统计。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct = 0
    total = 0
    
    for i in range(len(y_true) - 1):
        delta_true = y_true[i+1] - y_true[i]
        
        if y_true[i] == 0:
            continue
        relative_change = abs(delta_true) / abs(y_true[i])
        
        if relative_change <= threshold_percent:
            continue
        
        delta_pred = y_pred[i+1] - y_pred[i]
        
        if delta_true * delta_pred > 0:
            correct += 1
        
        total += 1
    
    if total == 0:
        return None
    
    return correct / total

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    
    y = df['runtime'].values
    
    # remove: runtime(label), job_id, run_time
    df.drop(['runtime','job_id','run_time'], axis=1, inplace=True)
    
    apps = df['app_name'].astype(str).values
    df.drop(['app_name'], axis=1, inplace=True)

    numeric_features = df.columns
    
    # OneHot for app_name
    encoder = OneHotEncoder(sparse=False)
    apps_encoded = encoder.fit_transform(apps.reshape(-1, 1))
    app_feature_names = encoder.get_feature_names_out(['app_name'])
    
    X = np.hstack([apps_encoded, df.values])
    feature_names = list(app_feature_names) + list(numeric_features)
    
    return X, y, feature_names, df

def prepare_subset(df, model_name):
        df2 = df.copy()
        df2.drop(['runtime','job_id','run_time'], axis=1, inplace=True)
        if model_name == 'placement':
            df2 = df2[['app_name', 'group_count']]
        if model_name == 'placement_gemm':
            df2 = df2[['app_name', 'group_count', 'gemm_min', 'gemm_mean', 'gemm_max']]
        if model_name == 'placement_gemm_allreduce':
            if MECHINE_NAME == "frontier":
                df2 = df2[['app_name', 'group_count', 'gemm_min', 'gemm_mean', 'gemm_max', 
                      'allreduce_1K', 'allreduce_1M', 'allreduce_16M', 'allreduce_2G']]
            else:
                df2 = df2[['app_name', 'group_count', 'gemm_min', 'gemm_mean', 'gemm_max', 
                      'allreduce_1K', 'allreduce_2K', 'allreduce_4K', 'allreduce_8K', 
                      'allreduce_16K', 'allreduce_32K', 'allreduce_64K', 'allreduce_128K', 
                      'allreduce_256K', 'allreduce_512K', 'allreduce_1M', 'allreduce_16M',
                      'allreduce_32M', 'allreduce_64M', 'allreduce_128M', 'allreduce_256M',
                      'allreduce_512M', 'allreduce_1G', 'allreduce_2G']]
        return df2

def train_test(model_name):
    csv_file = "combined_results_%s.csv" % MECHINE_NAME
    
    df_full = pd.read_csv(csv_file)

    df_amg = df_full[df_full['app_name'] == "AMG2023"].copy()
    df_deepcam = df_full[df_full['app_name'] == "deepCAM"].copy()
    df_nanogpt = df_full[df_full['app_name'] == "nanoGPT"].copy()
    df_milc = df_full[df_full['app_name'] == "MILC"].copy()
    
    test_size = 10
    test_size_milc = 7
    df_amg_test = df_amg.iloc[-test_size:].copy()
    df_nanogpt_test = df_nanogpt.iloc[-test_size:].copy()
    df_deepcam_test = df_deepcam.iloc[-test_size:].copy()
    df_milc_test = df_milc.iloc[-test_size_milc:].copy()
    df_train = pd.concat([df_amg.iloc[:-test_size], df_deepcam.iloc[:-test_size], df_nanogpt.iloc[:-test_size], df_milc.iloc[:-test_size]])
    # df_train = pd.concat([df_amg.iloc[:-test_size], df_deepcam.iloc[:-test_size], df_nanogpt.iloc[:-test_size]])

    # y_train / y_test
    y_train2 = df_train['runtime'].values
    y_amg_test2 = df_amg_test['runtime'].values
    y_nanogpt_test2 = df_nanogpt_test['runtime'].values
    y_deepcam_test2 = df_deepcam_test['runtime'].values
    y_milc_test2 = df_milc_test['runtime'].values
    # y_test2 = df_test['runtime'].values
    
    # job_id / run_time / runtime(目标) 不能做特征
    
    df_train2 = prepare_subset(df_train, model_name)
    print("=================================================================")
    # print(df_train2)
    df_amg_test2 = prepare_subset(df_amg_test, model_name)
    df_nanogpt_test2 = prepare_subset(df_nanogpt_test, model_name)
    df_deepcam_test2 = prepare_subset(df_deepcam_test, model_name)
    df_milc_test2 = prepare_subset(df_milc_test, model_name)
    # one-hot
    apps_train = df_train2['app_name'].astype(str).values

    apps_amg_test = df_amg_test2['app_name'].astype(str).values
    apps_nanogpt_test = df_nanogpt_test2['app_name'].astype(str).values
    apps_deepcam_test = df_deepcam_test2['app_name'].astype(str).values
    apps_milc_test = df_milc_test2['app_name'].astype(str).values
    df_train2.drop(['app_name'], axis=1, inplace=True)

    df_amg_test2.drop(['app_name'], axis=1, inplace=True)
    df_nanogpt_test2.drop(['app_name'], axis=1, inplace=True)
    df_deepcam_test2.drop(['app_name'], axis=1, inplace=True)
    df_milc_test2.drop(['app_name'], axis=1, inplace=True)
    
    enc = OneHotEncoder(sparse=False)
    
    apps_train_enc = enc.fit_transform(apps_train.reshape(-1, 1))
    apps_amg_test_enc = enc.transform(apps_amg_test.reshape(-1, 1))
    apps_nanogpt_test_enc = enc.transform(apps_nanogpt_test.reshape(-1, 1))
    apps_deepcam_test_enc = enc.transform(apps_deepcam_test.reshape(-1, 1))
    apps_milc_test_enc = enc.transform(apps_milc_test.reshape(-1, 1))
    
    X_train2 = np.hstack([apps_train_enc, df_train2.values])
    X_amg_test2 = np.hstack([apps_amg_test_enc, df_amg_test2.values])
    X_nanogpt_test2 = np.hstack([apps_nanogpt_test_enc, df_nanogpt_test2.values])
    X_deepcam_test2 = np.hstack([apps_deepcam_test_enc, df_deepcam_test2.values])
    X_milc_test2 = np.hstack([apps_milc_test_enc, df_milc_test2.values])
    
    feature_names2 = list(enc.get_feature_names_out(['app_name'])) + list(df_train2.columns)
    
    # XGBoost
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train2, y_train2)
    # print(X_train2[0])
    
    y_pred2 = model.predict(X_amg_test2)
    y_pred3 = model.predict(X_nanogpt_test2)
    y_pred4 = model.predict(X_deepcam_test2)
    y_pred5 = model.predict(X_milc_test2)
    
    print("\n=== %s AMG + deepCAM Results ===" % model_name)
    # print(y_test2)
    print(y_amg_test2)
    print(y_pred2)
    print(y_nanogpt_test2)
    print(y_pred3)
    print(y_deepcam_test2)
    print(y_pred4)
    print(y_milc_test2)
    print(y_pred5)
    importances_xgb = model.feature_importances_
    sorted_idx_xgb = np.argsort(importances_xgb)[::-1]
    # Save sorted feature importances to CSV
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_names2[i] for i in sorted_idx_xgb],
        'Importance': [importances_xgb[i] for i in sorted_idx_xgb]
    })
    feature_importance_df.to_csv('%s_feature_importance_%s.csv' % (MECHINE_NAME, model_name), index=False)
    
    plt.figure()
    plt.plot(range(1,test_size+1), y_amg_test2, marker='o', label="AMG Actual Runtime")
    plt.plot(range(1,test_size+1), y_pred2, marker='x', label="AMG Predicted Runtime")
    plt.plot(range(test_size+1,2*test_size+1), y_nanogpt_test2, marker='s', label="nanoGPT Actual Runtime")
    plt.plot(range(test_size+1,2*test_size+1), y_pred3, marker='*', label="nanoGPT Predicted Runtime")
    plt.plot(range(2*test_size+1,3*test_size+1), y_deepcam_test2, marker='s', label="deepCAM Actual Runtime")
    plt.plot(range(2*test_size+1,3*test_size+1), y_pred4, marker='s', label="deepCAM Predicted Runtime")
    plt.plot(range(3*test_size+1,3*test_size+test_size_milc+1), y_milc_test2, marker='s', label="MILC Actual Runtime")
    plt.plot(range(3*test_size+1,3*test_size+test_size_milc+1), y_pred5, marker='s', label="MILC Predicted Runtime")
    plt.xlabel("Test Sample Index (Last %d)" % test_size)
    plt.ylabel("Runtime")
    plt.title("Last %d Points Actual vs Predicted" % test_size)
    plt.legend()
    plt.savefig("%s_%s_amg_deepcam_nanoGPT_test%d_prediction.png" % (MECHINE_NAME, model_name, test_size), bbox_inches='tight', dpi=300)
    plt.savefig("%s_%s_amg_deepcam_nanoGPT_test%d_prediction.pdf" % (MECHINE_NAME, model_name, test_size), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\n[INFO] Done. Check shap_summary_xgb.png for SHAP summary, and %s_amg_deepcam_nanoGPT_test%d_prediction.png for line plot." % (model_name, test_size))

    mape_amg = mean_absolute_percentage_error(y_amg_test2, y_pred2)
    mape_nanogpt = mean_absolute_percentage_error(y_nanogpt_test2, y_pred3)
    mape_deepcam = mean_absolute_percentage_error(y_deepcam_test2, y_pred4)
    mape_milc = mean_absolute_percentage_error(y_milc_test2, y_pred5)
    print("\n--- MAPE (Mean Absolute Percentage Error) ---")
    print(f"{model_name} AMG Test MAPE: {mape_amg:.2f}%")
    print(f"{model_name} nanoGPT Test MAPE: {mape_nanogpt:.2f}%")
    print(f"{model_name} deepCAM Test MAPE: {mape_deepcam:.2f}%")
    print(f"{model_name} MILC Test MAPE: {mape_milc:.2f}%")
    da_amg = direction_accuracy_threshold(y_amg_test2, y_pred2, threshold_percent=0.02)
    da_nanogpt = direction_accuracy_threshold(y_nanogpt_test2, y_pred3, threshold_percent=0.02)
    da_deepcam = direction_accuracy_threshold(y_deepcam_test2, y_pred4, threshold_percent=0.02)
    da_milc = direction_accuracy_threshold(y_milc_test2, y_pred5, threshold_percent=0.02)
    print("\n--- Direction Accuracy (with ±2% threshold) ---")
    if da_amg is not None:
        print(f"{model_name} AMG Test Direction Accuracy: {da_amg:.2f}")
    else:
        print(f"{model_name} AMG Test Direction Accuracy: None (no changes exceeded ±2%)")
    
    if da_nanogpt is not None:
        print(f"{model_name} nanoGPT Test Direction Accuracy: {da_nanogpt:.2f}")
    else:
        print(f"{model_name} nanoGPT Test Direction Accuracy: None (no changes exceeded ±2%)")
    
    if da_deepcam is not None:
        print(f"{model_name} deepCAM Test Direction Accuracy: {da_deepcam:.2f}")
    else:
        print(f"{model_name} deepCAM Test Direction Accuracy: None (no changes exceeded ±2%)")
    
    if da_milc is not None:
        print(f"{model_name} MILC Test Direction Accuracy: {da_milc:.2f}")
    else:
        print(f"{model_name} MILC Test Direction Accuracy: None (no changes exceeded ±2%)")

if __name__ == "__main__":
    # train_test('placement')
    # train_test('placement_gemm')
    # train_test('placement_gemm_allreduce')
    train_test('full')

