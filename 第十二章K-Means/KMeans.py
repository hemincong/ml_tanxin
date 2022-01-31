# 导入相应的库
from copy import deepcopy

from numpy import reshape
from pylab import imread, imshow
from sklearn.cluster import KMeans

# 读入图片数据
img = imread('/home/anaconda/data/Z_NLP/sample.jpg')  # img: 图片的数据
# 把三维矩阵转换成二维的矩阵
pixel = reshape(img, (img.shape[0] * img.shape[1], 3))
pixel_new = deepcopy(pixel)
print(img.shape)
# 创建k-means模型, 可以试着改一下 n_clusters参数试试
model = KMeans(n_clusters=3)
labels = model.fit_predict(pixel)
palette = model.cluster_centers_
for i in range(len(pixel)):
    pixel_new[i, :] = palette[labels[i]]
# 展示重新构造的图片(压缩后的图片)
imshow(reshape(pixel_new, (img.shape[0], img.shape[1], 3)))
