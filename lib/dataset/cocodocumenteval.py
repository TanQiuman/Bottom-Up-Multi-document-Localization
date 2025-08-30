import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import copy
from pycocotools.cocoeval import COCOeval

class COCODocumenteval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)
        # 如果是 4 个关键点，可以重新定义这里的参数
        if self.params.iouType == 'keypoints':
            new_sigma_values_for_4_points= np.array(
                [0.2,0.2,0.2,0.2])
            self.params.kpt_oks_sigmas = np.array([new_sigma_values_for_4_points])

    # 如果需要，可以在这里添加新的方法或覆盖父类方法以适应特定需求
    def new_method_for_document(self):
        print("This is a new method specific to COCODocumenteval.")

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        # 修改为四个关键点的标准差数组
        sigmas = np.array([.26, .25, .25, .35])/10
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        for j, gt in enumerate(gts):
            g = np.array(gt['keypoints'])
            # 确保按照四个关键点进行处理
            xg = g[0::3][:4];
            yg = g[1::3][:4];
            vg = g[2::3][:4]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2];
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3];
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                # 确保按照四个关键点进行处理
                xd = d[0::3][:4];
                yd = d[1::3][:4]
                if k1 > 0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious
