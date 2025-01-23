import math
import config
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR

def lambda_step(step):
    max_lr = config.max_lr
    base_lr = config.learning_rate #4e-5
    
    steps_per_epoch = config.steps_per_epoch
    total_steps = config.max_epoch * steps_per_epoch
    max_step = config.pct_start * steps_per_epoch
    total_cosine_steps = total_steps - max_step
    
    if step < max_step:
        # 초기 학습률에서 max_lr까지 선형 증가
        return 1 + (max_lr / base_lr - 1) * (step / max_step)
    else:
        # max_lr에서 base_lr까지 코사인 함수로 감소
        cosine_steps = step - max_step
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_steps / total_cosine_steps))
        return 1 + (max_lr / base_lr - 1) * cosine_decay


def create_custom_lambda_scheduler(optimizer, base_lr=1e-4, max_lr=0.001, final_lr=1e-6,
                                   total_epochs=250, steps_per_epoch=23,
                                   max_epoch=10, decay_rate=None):
    """
    커스텀 LambdaLR 스케줄러를 생성합니다.
    
    Parameters:
    - optimizer: PyTorch 옵티마이저
    - base_lr: 초기 학습률
    - max_lr: 최대 학습률
    - final_lr: 최종 학습률 (총 학습 후 도달)
    - total_epochs: 전체 학습 에포크 수
    - steps_per_epoch: 에포크당 스텝 수
    - max_epoch: 최대 학습률에 도달할 에포크 수
    - decay_rate: 지수적 감소율 (None이면 자동 계산)
    
    Returns:
    - LambdaLR 스케줄러
    """
    total_steps = total_epochs * steps_per_epoch
    max_step = max_epoch * steps_per_epoch

    if decay_rate is None:
        # 자동으로 decay_rate 계산
        # final_lr = max_lr * exp(-decay_rate * (total_steps - max_step))
        decay_rate = math.log(max_lr / final_lr) / (total_steps - max_step)
    
    def lr_lambda(step):
        if step < max_step:
            # 선형 증가: base_lr -> max_lr
            return 1 + (max_lr / base_lr - 1) * (step / max_step)
        else:
            # 지수적 감소: max_lr -> final_lr
            return (max_lr / base_lr) * math.exp(-decay_rate * (step - max_step))
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)