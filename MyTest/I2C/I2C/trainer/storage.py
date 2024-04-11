import numpy as np

class Buffer(object):
    def __init__(self,arglist, obs_input_size, obs_int_input_size):
        self.num_steps = int(arglist.prior_buffer_size)
        self.obs_inputs = np.zeros((self.num_steps, int(obs_input_size)))
        self.obs_int_inputs = np.zeros((self.num_steps, int(obs_int_input_size)))
        self.KL_values = np.zeros((self.num_steps, 1))
        self.labels = np.zeros((self.num_steps, 2))
        self.step = 0
        self.threshold = 0
        self.percentile = arglist.prior_training_percentile
        self.pos_idx = []
        self.neg_idx = []
        self.pos_t = 0
        self.neg_t = 0


    def insert(self, size, obs_inputs, obs_int_inputs, KL_values):
        if self.step+size <= self.num_steps:
            self.obs_inputs[self.step:self.step+size] = obs_inputs
            self.obs_int_inputs[self.step:self.step+size] = obs_int_inputs
            self.KL_values[self.step:self.step+size] = KL_values[:,None]
            self.step = self.step + size
            return False
        else:
            step_tmp = self.num_steps - self.step
            self.obs_inputs[self.step:] = obs_inputs[0:step_tmp]
            self.obs_int_inputs[self.step:] = obs_int_inputs[0:step_tmp]
            self.KL_values[self.step:] = KL_values[0:step_tmp,None]
            self.step = 0
            self.get_labels()
            self.get_shuffle()
            return True

    def get_threshold(self):
        self.threshold = np.percentile(self.KL_values, self.percentile)
        return self.threshold

    def get_labels(self):
        self.get_threshold()
        for i in range(self.num_steps):
            if self.KL_values[i] > self.threshold:
                self.labels[i] = [1,0]
                self.pos_idx.append(i)
            elif self.KL_values[i] <= self.threshold:
                self.labels[i] = [0,1]
                self.neg_idx.append(i)

    def get_shuffle(self):
        np.random.shuffle(self.pos_idx)
        np.random.shuffle(self.neg_idx)

    def get_samples(self, batch_size):
        if (self.pos_t + int(batch_size/2)) < len(self.pos_idx):
            pos_samples = self.pos_idx[self.pos_t:self.pos_t + int(batch_size/2)]
            self.pos_t += int(batch_size/2)
        else:
            pos_samples = self.pos_idx[self.pos_t:]
            self.pos_t = 0
        if (self.neg_t + int(batch_size/2)) < len(self.neg_idx):
            neg_samples = self.neg_idx[self.neg_t:self.neg_t + int(batch_size/2)]
            self.neg_t += int(batch_size/2)
        else:
            neg_samples = self.neg_idx[self.neg_t:]
            self.neg_t = 0
        
        all_samples = pos_samples + neg_samples
        return self.obs_inputs[np.array(all_samples)], self.obs_int_inputs[np.array(all_samples)], self.labels[np.array(all_samples)]





    