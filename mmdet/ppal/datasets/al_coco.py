from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class ALCocoDataset(CocoDataset):
    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['cls_uncertainty'] = float(bboxes[i][5])
                    data['box_uncertainty'] = float(bboxes[i][6])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results