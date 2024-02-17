from pathlib import Path

from shock.utils import h5Dataset, preprocessing_cnt

savePath = Path("path/to/your/save/path")
rawDataPath = Path("path/to/your/raw/data/path")
group = rawDataPath.glob("*.cnt")

# 预处理参数
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

# chunks参数, 1s 数据
chunks = (62, rsfreq)

dataset = h5Dataset(savePath, "dataset")
for cntFile in group:
    print(f"processing {cntFile.name}")
    eegData, chOrder = preprocessing_cnt(cntFile, l_freq, h_freq, rsfreq)
    chOrder = [s.upper() for s in chOrder]
    eegData = eegData[:, : -10 * rsfreq]
    grp = dataset.addGroup(grpName=cntFile.stem)
    dset = dataset.addDataset(grp, "eeg", eegData, chunks)

    # dataset属性，预处理信息
    dataset.addAttributes(dset, "lFreq", l_freq)
    dataset.addAttributes(dset, "hFreq", h_freq)
    dataset.addAttributes(dset, "rsFreq", rsfreq)
    dataset.addAttributes(dset, "chOrder", chOrder)

dataset.save()
