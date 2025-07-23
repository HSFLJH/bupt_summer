# 工具函数
def collate_fn(batch):
    return tuple(zip(*batch))
