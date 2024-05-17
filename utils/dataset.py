from torch.utils.data import Dataset

class DynamicalSystemDataset(Dataset):
    def __init__(self, data, step_num = 1, step_interval=1):
        """
        Parameters
        ----------
        data : np.array(軌道数×ステップ数×システムの次元)
            力学系の時系列データ
        step_num : int
            力学系のステップ数
        step_interval : int
            力学系の時間間隔
        """
        self.data = data
        self.step_num = step_num
        self.step_interval = step_interval
        self.num_trajectory = self.data.shape[0]
        self.start_step_num = self.data.shape[1] - (step_num - 1) * step_interval
        assert self.start_step_num > 0 
        self.len = self.num_trajectory * self.start_step_num

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        traj_i = int(idx/self.start_step_num)
        start_step = idx%self.start_step_num
        end_step = start_step + (self.step_num - 1) * self.step_interval + 1
        return self.data[traj_i,start_step:end_step:self.step_interval,:]