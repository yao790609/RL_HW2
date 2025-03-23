from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# 動作方向（上、下、左、右）
ACTIONS = {'↑': (-1, 0), '↓': (1, 0), '←': (0, -1), '→': (0, 1)}
ACTION_LIST = ['↑', '↓', '←', '→']

# Value Iteration 參數
GAMMA = 0.9  # 折扣因子
THETA = 1e-4  # 收斂條件

def value_iteration(n, start, end, obstacles):
    """執行 Value Iteration 來計算最優策略"""
    # 初始化價值矩陣
    value_matrix = np.zeros((n, n))
    
    # 設定終點的獎勵
    rewards = np.full((n, n), -0.5)  # 普通移動的 reward = -0.5
    rewards[end] = 20  # 碰到終點 reward = 20
    for obs in obstacles:
        rewards[obs] = -10  # 提高障礙物的懲罰，使智能體更有動力避開它們
    
    # 初始化政策矩陣（稍後會根據價值函數更新）
    policy_matrix = np.full((n, n), " ", dtype=object)
    
    # Value Iteration 算法
    iteration = 0
    max_iterations = 1000  # 防止永遠循環
    
    while True:
        delta = 0
        iteration += 1
        for i in range(n):
            for j in range(n):
                if (i, j) == end:
                    value_matrix[i, j] = rewards[end]  # 確保終點的值始終是獎勵
                    continue
                if (i, j) in obstacles:
                    value_matrix[i, j] = rewards[i, j]  # 障礙物的值
                    continue
                
                old_value = value_matrix[i, j]
                
                # 計算每個動作的預期價值
                action_values = []
                for action in ACTION_LIST:
                    di, dj = ACTIONS[action]
                    ni, nj = max(0, min(n-1, i+di)), max(0, min(n-1, j+dj))
                    
                    # 如果移動會導致碰撞障礙物，則停留在原地
                    if (ni, nj) in obstacles:
                        ni, nj = i, j
                    
                    # 如果下一步是終點，給予額外獎勵
                    if (ni, nj) == end:
                        action_values.append(rewards[i, j] + GAMMA * (value_matrix[ni, nj] + 5))
                    else:
                        action_values.append(rewards[i, j] + GAMMA * value_matrix[ni, nj])
                
                # 取最大價值更新
                value_matrix[i, j] = max(action_values)
                
                # 計算差值用於收斂判斷
                delta = max(delta, abs(old_value - value_matrix[i, j]))
        
        # 檢查收斂條件或最大迭代次數
        if delta < THETA or iteration >= max_iterations:
            break
    
    # 根據最終的價值函數導出最佳策略
    for i in range(n):
        for j in range(n):
            if (i, j) == end:
                policy_matrix[i, j] = "E"  # 終點標記
                continue
            if (i, j) in obstacles:
                policy_matrix[i, j] = "X"  # 障礙物標記
                continue
                
            # 為起點添加特殊處理，確保起點有明確的方向指向終點
            if (i, j) == start:
                # 根據起點和終點的相對位置選擇方向
                best_action = get_direction_to_end(i, j, end, obstacles, n)
                policy_matrix[i, j] = best_action
                continue
                
            best_action = None
            best_value = float('-inf')
            for action in ACTION_LIST:
                di, dj = ACTIONS[action]
                ni, nj = max(0, min(n-1, i+di)), max(0, min(n-1, j+dj))
                
                # 如果移動會導致碰撞障礙物，則停留在原地
                if (ni, nj) in obstacles:
                    ni, nj = i, j
                    
                value = rewards[i, j] + GAMMA * value_matrix[ni, nj]
                if value > best_value:
                    best_value = value
                    best_action = action
            
            policy_matrix[i, j] = best_action

    return value_matrix.tolist(), policy_matrix.tolist()

def get_direction_to_end(i, j, end, obstacles, n):
    """計算從位置(i,j)到達終點的最佳方向"""
    end_i, end_j = end
    best_action = None
    
    # 計算曼哈頓距離
    if abs(i - end_i) > abs(j - end_j):
        # 垂直距離更大，優先考慮垂直移動
        if i < end_i:
            best_action = '↓'
        else:
            best_action = '↑'
    else:
        # 水平距離更大或相等，優先考慮水平移動
        if j < end_j:
            best_action = '→'
        else:
            best_action = '←'
    
    # 檢查選定的方向是否會導致碰撞障礙物
    di, dj = ACTIONS[best_action]
    ni, nj = max(0, min(n-1, i+di)), max(0, min(n-1, j+dj))
    
    if (ni, nj) in obstacles:
        # 如果會碰撞，選擇另一個方向
        possible_actions = ['↑', '↓', '←', '→']
        possible_actions.remove(best_action)
        
        for action in possible_actions:
            di, dj = ACTIONS[action]
            ni, nj = max(0, min(n-1, i+di)), max(0, min(n-1, j+dj))
            if (ni, nj) not in obstacles:
                return action
    
    return best_action

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('gridworld.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.json
    n = data['n']
    start = tuple(data['start'])
    end = tuple(data['end'])
    obstacles = [tuple(x) for x in data['obstacles']]

    value_matrix, policy_matrix = value_iteration(n, start, end, obstacles)
    
    return jsonify({'value_matrix': value_matrix, 'policy_matrix': policy_matrix})

if __name__ == '__main__':
    app.run(debug=True)