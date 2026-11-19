"""Keep RobustKT on its verified length200 MPS path, preserving causal outputs."""
from torch.nn import functional as F
from check_attention_model_readiness import module


def robustkt_class(device):
    original = module('robustkt', device).Robustkt

    class Length200Robustkt(original):
        def forward(self, q_data, target, pid_data=None, qtest=False):
            length = q_data.shape[1]
            assert 1 <= length <= 200
            assert self.n_pid == 0, 'This portability adapter is concept-only'
            if length < 200:
                q_data = F.pad(q_data, (0, 200-length))
                target = F.pad(target, (0, 200-length))
                if pid_data is not None:
                    pid_data = F.pad(pid_data, (0, 200-length))
            result = super().forward(q_data, target, pid_data, qtest)
            if qtest:
                return result[0][:,:length], result[1], result[2][:,:length]
            return result[0][:,:length], result[1]

    return Length200Robustkt
