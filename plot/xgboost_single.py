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

def train_test(model_name, app_name):
    csv_file = "combined_results_%s.csv" % MECHINE_NAME
    
    df_full = pd.read_csv(csv_file)

    df_app = df_full[df_full['app_name'] == app_name].copy()
    
    test_size = 10
    df_app_test = df_app.iloc[-test_size:].copy()
    
    df_train = pd.concat([df_app.iloc[:-test_size]])
    # df_test = pd.concat([df_amg_test, df_nanogpt_test])

    # y_train / y_test
    y_train2 = df_train['runtime'].values
    y_app_test2 = df_app_test['runtime'].values
    # y_test2 = df_test['runtime'].values
    
    # job_id / run_time / runtime(目标) 不能做特征
    
    df_train2 = prepare_subset(df_train, model_name)
    print("=================================================================")
    print(df_train2)
    df_app_test2 = prepare_subset(df_app_test, model_name)
    # one-hot
    apps_train = df_train2['app_name'].astype(str).values

    apps_app_test = df_app_test2['app_name'].astype(str).values
    df_train2.drop(['app_name'], axis=1, inplace=True)

    df_app_test2.drop(['app_name'], axis=1, inplace=True)
    
    enc = OneHotEncoder(sparse=False)
    apps_train_enc = enc.fit_transform(apps_train.reshape(-1, 1))
    apps_app_test_enc = enc.transform(apps_app_test.reshape(-1, 1))
    
    X_train2 = np.hstack([apps_train_enc, df_train2.values])
    X_app_test2 = np.hstack([apps_app_test_enc, df_app_test2.values])
    feature_names2 = list(enc.get_feature_names_out(['app_name'])) + list(df_train2.columns)
    
    # print(X_amg_test2)
    # print(feature_names2)
    
    # XGBoost
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train2, y_train2)
    
    y_pred2 = model.predict(X_app_test2)
    
    print("\n=== %s AMG + deepCAM Results ===" % model_name)
    # print(y_test2)
    print(y_app_test2)
    print(y_pred2)
    importances_xgb = model.feature_importances_
    sorted_idx_xgb = np.argsort(importances_xgb)[::-1]
    # Save sorted feature importances to CSV
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_names2[i] for i in sorted_idx_xgb],
        'Importance': [importances_xgb[i] for i in sorted_idx_xgb]
    })
    feature_importance_df.to_csv('single_test_%s_feature_importance_%s_%s.csv' % (MECHINE_NAME, model_name, app_name), index=False)
    
    plt.figure()
    plt.plot(range(1,test_size+1), y_app_test2, marker='o', label="Actual Runtime")
    plt.plot(range(1,test_size+1), y_pred2, marker='x', label="Predicted Runtime")
    plt.xlabel("Test Sample Index (Last %d)" % test_size)
    plt.ylabel("Runtime")
    plt.title("Last %d Points Actual vs Predicted" % test_size)
    plt.legend()
    plt.savefig("single_test_%s_%s_%s_test%d_prediction.png" % (MECHINE_NAME, model_name, app_name, test_size), bbox_inches='tight', dpi=300)
    plt.savefig("single_test_%s_%s_%s_test%d_prediction.pdf" % (MECHINE_NAME, model_name, app_name, test_size), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\n[INFO] Done. Check shap_summary_xgb.png for SHAP summary, and %s_amg_deepcam_nanoGPT_test%d_prediction.png for line plot." % (model_name, test_size))

    mape_app = mean_absolute_percentage_error(y_app_test2, y_pred2)
    print("\n--- MAPE (Mean Absolute Percentage Error) ---")
    print("MAPE: %s" % mape_app)
    da_app = direction_accuracy_threshold(y_app_test2, y_pred2, threshold_percent=0.02)
    print("\n--- Direction Accuracy (with ±2% threshold) ---")
    if da_app is not None:
        print("Direction Accuracy: %s" % da_app)
    else:
        print("Direction Accuracy: None (no changes exceeded ±2%)")

if __name__ == "__main__":
    # train_test('placement', 'AMG2023')
    # train_test('placement_gemm', 'AMG2023')
    # train_test('placement_gemm_allreduce', 'AMG2023')
    # train_test('full', 'AMG2023')
    
    # train_test('placement', 'deepCAM')
    # train_test('placement_gemm', 'deepCAM')
    # train_test('placement_gemm_allreduce', 'deepCAM')
    train_test('full', 'deepCAM')
    
    # train_test('placement', 'nanoGPT')
    # train_test('placement_gemm', 'nanoGPT')
    # train_test('placement_gemm_allreduce', 'nanoGPT')
    # train_test('full', 'nanoGPT')
