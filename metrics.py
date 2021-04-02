from typing import Tuple

import torch


def scoring(pred_y: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    """n번 추론 중 정답 추론이 몇 개인지 알려주는 함수

    Args:
        pred_y (torch.Tensor): 추론 값
        y (torch.Tensor): 타겟 값
    
    Returns:
        int: 정답 추론 수
        int: 추론 횟수
    """

    with torch.no_grad():
        pred_y = pred_y.max(dim=-1)[1]
        scoring = (pred_y == y).float()
        correct_num = scoring.sum().to(torch.device('cpu')).numpy()
        total_num = len(scoring)
    return correct_num, total_num