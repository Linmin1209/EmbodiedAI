def _compute_single_navigation_score(info, baseline=None, ne_max=10.0):
    """
    info 是字典格式，示例：
    {'NE': [[7.5569363, 0]],
     'SPL': [[0, 0]],
     'PL': ([3.874446], [6.1169990953058]),
     'stage_success': [[False, False]]}

    baseline 可以用来做进一步调整
    ne_max 是最大导航误差阈值，超过视为0分
    """

    # 取第一个stage的数据
    ne_list = info.get('NE', [[0, 0]])[0]
    spl_list = info.get('SPL', [[0, 0]])[0]
    pl_actual = info.get('PL', ([0], [1]))[0][0]
    pl_ideal = info.get('PL', ([0], [1]))[1][0]
    success_list = info.get('stage_success', [[False, False]])[0]

    # 取第一个阶段成功状态（你可以根据需求调整）
    success = any(success_list)

    # 计算NE分数（用第一个NE）
    ne = ne_list[0]
    ne_score = max(0, 1 - ne / ne_max) * 100

    # 计算PL分数
    if pl_actual > 0:
        pl_score = min(1, pl_ideal / pl_actual) * 100
    else:
        pl_score = 0

    # SPL取第一个
    spl = spl_list[0]
    spl_score = spl * 100

    # 总分计算
    if success:
        total_score = 50 + 0.3 * ne_score + 0.2 * pl_score
    else:
        # total_score = 0  # 或者你也可以给失败打个基础分
        total_score = 0.3 * ne_score + 0.2 * pl_score

    # 限制总分 0~100
    total_score = max(0, min(100, total_score))

    return total_score

def compute_average_navigation_score(info_list, baseline=None, ne_max=10.0):
    scores = []
    for info in info_list:
        scores.append(_compute_single_navigation_score(info, baseline, ne_max))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

# 示例用法
info_example = {
    'NE': [[5.7403917, 0]],
    'SPL': [[0, 0]],
    'PL': ([1.4445728], [9.54404]),
    'stage_success': [[False, False]]
}

info_example_2 = {
    'NE': [[7.764868, 0]],
    'SPL': [[0, 0]],
    'PL': ([4.107266], [8.11208]),
    'stage_success': [[False, False]]
}


# info_example_2 = {
#     'NE': [[2.0, 0]],
#     'SPL': [[0.5, 0]],
#     'PL': ([5.0], [5.0]),
#     'stage_success': [[True, True]]
# }

info_list_example = [info_example, info_example_2]

score = compute_average_navigation_score(info_list_example)
print(f"平均得分: {score:.2f}")
