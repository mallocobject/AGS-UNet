# DDU-Net
config.yaml配置GPU_id和GPU_num

scripts关注噪声类别以及强度

noise_type可取"bw", "em", "ma", "emb"

SNR可取-4, -2, 0, 2, 4

详情见run.py

mode为train即为训练，完成后切换至test评测指标(utils/metrics.py)
