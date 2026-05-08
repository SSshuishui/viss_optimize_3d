1、需求：生成包含河外点源catalog，点源的自吸收效应，银河系辐射的小尺度结构，自由自由辐射小尺度结构的天图
2、辨率NSIDE=4096，对应resolution~0.86 arcmin。
3、文件格式：.hdf5文件
4、文件大小：约3Gb（single）
5、数据介绍与读取步骤：
```
import numpy as np
import h5py
import healpy as hp
import matplotlib.pyplot as plt

with h5py.File('1.0MHz_with_absorption.hdf5','r') as fe:
	print('fe.keys()',fe.keys())
    skymap = fe['skymap'][:]
    extragalactic_point_source_catalog_intrinsic = fe['extragalactic_point_source_catalog_intrinsic'][:] 
    print('skymap',skymap)
	
hp.mollview(np.log10(skymap),cmap = plt.cm.jet,min=6.7,max=8.4)
hp.mollview(extragalactic_point_source_catalog_intrinsic,norm='hist',cmap = plt.cm.jet)
plt.show()
plt.close()
```
其中，skymap为天图数据，extragalactic_point_source_catalog_intrinsic为河外点源intrinsic亮度。