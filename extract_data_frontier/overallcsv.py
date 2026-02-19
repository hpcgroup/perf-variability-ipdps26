import os
import pandas as pd
import util

def main():
    base_dir = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar")
    apps = ['AMG2023', 'MILC', 'nanoGPT', 'deepcam']
    nodes = 64

    df = pd.DataFrame(columns=['app_name', 'date', 'job_id', 'runtime_seconds'])

    timestamp_file = {
        "AMG2023": "output-AMG2023.log",
        "deepcam": "output-deepcam.log",
        "nanoGPT": "output-nanoGPT.log",
        "MILC": "output-gemm.log",
    }

    job_dirs = []
    for app in apps:
        app_dir = os.path.join(base_dir, f'{app}_logs', f'{nodes}nodes')

        job_dirs = job_dirs + sorted(
            [(app, os.path.join(app_dir, d)) for d in os.listdir(app_dir) if util.verify_app_dir(d, app, nodes)],
            key=lambda x: x[1],
        )
        
        if app == 'AMG2023':
            app_dir = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar/AMG2023_logs/64_nodes_before1.15")
            job_dirs = job_dirs + sorted(
                [(app, os.path.join(app_dir, d)) for d in os.listdir(app_dir) if util.verify_app_dir(d, app, nodes)],
                key=lambda x: x[1],
            )
        if app == 'deepcam':
            app_dir = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar/deepcam_logs/64nodes_no_RDZV_env")
            job_dirs = job_dirs + sorted(
                [(app, os.path.join(app_dir, d)) for d in os.listdir(app_dir) if util.verify_app_dir(d, app, nodes)],
                key=lambda x: x[1],
            )
        if app == 'nanoGPT':
            app_dir = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar/nanoGPT_logs/64nodes_no_RDZV_env")
            job_dirs = job_dirs + sorted(
                [(app, os.path.join(app_dir, d)) for d in os.listdir(app_dir) if util.verify_app_dir(d, app, nodes)],
                key=lambda x: x[1],
            )
            app_dir = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar/nanoGPT_logs/64nodes_notorchprof")
            job_dirs = job_dirs + sorted(
                [(app, os.path.join(app_dir, d)) for d in os.listdir(app_dir) if util.verify_app_dir(d, app, nodes)],
                key=lambda x: x[1],
            )

    for app, job_dir in job_dirs:
        df.loc[df.shape[0]] = [
            app, 
            util.parse_timestamp(os.path.join(job_dir, timestamp_file[app]), app if app != 'MILC' else 'gemm'),
            util.parse_job_id(os.path.basename(job_dir), app, nodes),
            util.parse_app_time(job_dir, app, nodes)
        ]

    df = df.dropna()
    df = df.rename({
        'date': 'run_time',
        'runtime_seconds': 'runtime',
    }, axis=1)
    df.to_csv('overall_runtime.csv', index=False)
        

if __name__ == "__main__":
    main()