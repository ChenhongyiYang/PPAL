import json
import numpy as np


from mmdet.ppal.builder import SAMPLER
from mmdet.ppal.sampler.al_sampler_base import BaseALSampler
from mmdet.ppal.utils.running_checks import sys_echo


eps = 1e-10


@SAMPLER.register_module()
class DiversitySampler(BaseALSampler):
    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        dataset_type,
    ):
        super(DiversitySampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.log_init_info()

    @staticmethod
    def k_centroid_greedy(dis_matrix, K):
        N = dis_matrix.shape[0]
        centroids = []
        c = np.random.randint(0, N, (1,))[0]
        centroids.append(c)
        i = 1
        while i < K:
            centroids_diss = dis_matrix[:, centroids].copy()
            centroids_diss = centroids_diss.min(axis=1)
            centroids_diss[centroids] = -1
            new_c = np.argmax(centroids_diss)
            centroids.append(new_c)
            i += 1
        return centroids

    @staticmethod
    def kmeans(dis_matrix, K, n_iter=100):
        N = dis_matrix.shape[0]
        centroids = DiversitySampler.k_centroid_greedy(dis_matrix, K)
        data_indices = np.arange(N)

        assign_dis_records = []
        for _ in range(n_iter):
            centroid_dis = dis_matrix[:, centroids]
            cluster_assign = np.argmin(centroid_dis, axis=1)
            assign_dis = centroid_dis.min(axis=1).sum()
            assign_dis_records.append(assign_dis)

            new_centroids = []
            for i in range(K):
                cluster_i = data_indices[cluster_assign == i]
                assert len(cluster_i) >= 1
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
                new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
                new_centroids.append(new_centroid_i)
            centroids = np.array(new_centroids)
        return centroids.tolist()


    def al_acquisition(self, image_dis_path, last_label_path):
        with open(image_dis_path, 'rb') as frb:
            image_dis_matrix = np.load(frb)
            image_ids = np.load(frb).reshape(-1)

        centroids = DiversitySampler.kmeans(image_dis_matrix, K=self.n_images)

        with open(last_label_path) as f:
            results = json.load(f)

        last_labeled_img_ids = [x['id'] for x in results['images']]
        image_hit = dict()
        for img_id in self.oracle_data.keys():
            image_hit[img_id] = 0
        for img_id in last_labeled_img_ids:
            image_hit[img_id] = 1

        rest_image_ids = []
        for img_id in self.oracle_data.keys():
            if image_hit[img_id] == 0:
                rest_image_ids.append(img_id)

        sampled_img_ids = image_ids[centroids].tolist()
        for img_id in sampled_img_ids:
            rest_image_ids.remove(img_id)
        unsampled_img_ids = rest_image_ids

        return sampled_img_ids, unsampled_img_ids

    def al_round(self, result_path, image_dis_path, last_label_path, out_label_path, out_unlabeled_path):
        sys_echo('\n\n>> Starting Active Learning Acquisition!!!')
        self.round += 1
        self.log_info(result_path, image_dis_path, out_label_path, out_unlabeled_path)
        self.latest_labeled = last_label_path
        sampled_img_ids, rest_img_ids = self.al_acquisition(image_dis_path, last_label_path)
        self.create_jsons(sampled_img_ids, rest_img_ids, last_label_path, out_label_path, out_unlabeled_path)
        sys_echo('>> Active Learning Acquisition Complete!!!\n\n')

    def log_info(self, result_path, image_dis_path, out_label_path, out_unlabeled_path):
        sys_echo('>>>> Round: %d' % self.round)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation Path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Sampled images per round: %d (%.2f%%)' % (self.n_images, 100. * float(self.n_images) / self.image_pool_size))
        sys_echo('>>>> Unlabeled results path: %s' % result_path)
        sys_echo('>>>> Image distance cache: %s' % image_dis_path)
        sys_echo('>>>> Output label file path: %s' % out_label_path)
        sys_echo('>>>> Output unlabeled image info path: %s' % out_unlabeled_path)

    def log_init_info(self):
        sys_echo('>> %s initialized:'%self.__class__.__name__)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Sampled image per round: %d (%.2f%%)' % (self.n_images, 100. * float(self.n_images) / self.image_pool_size))
        sys_echo('\n')

