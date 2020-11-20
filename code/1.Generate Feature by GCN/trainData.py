from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index):
        # return (self.data_set['dd'], self.data_set['gg'],
        #         self.data_set['gd']['train'], None,
        #         self.data_set['gd_p'], self.data_set['gd_true'])

        return (self.data_set['mm'], self.data_set['gg'],
                self.data_set['gm']['train'], None,
                self.data_set['gm_p'], self.data_set['gm_true'])

    def __len__(self):
        return self.nums



