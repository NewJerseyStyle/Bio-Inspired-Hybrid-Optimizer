import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        a = x
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        return self.softmax(z)
    
    def get_params(self):
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.flatten())
            params.append(b.flatten())
        return np.concatenate(params)
    
    def set_params(self, params):
        idx = 0
        for i in range(len(self.weights)):
            w_shape, b_shape = self.weights[i].shape, self.biases[i].shape
            w_size, b_size = np.prod(w_shape), np.prod(b_shape)
            self.weights[i] = params[idx:idx+w_size].reshape(w_shape)
            idx += w_size
            self.biases[i] = params[idx:idx+b_size].reshape(b_shape)
            idx += b_size
    
    def evaluate(self, x, y):
        pred = self.forward(x)
        return np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
    
    def compute_loss(self, x, y):
        pred = self.forward(x)
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return -np.mean(np.sum(y * np.log(pred), axis=1))
    
    def backward(self, x, y):
        m = x.shape[0]
        activations = [x]
        z_values = []
        
        a = x
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.relu(z)
            activations.append(a)
        
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        output = self.softmax(z)
        
        dz = output - y
        gradients_w, gradients_b = [], []
        
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T)
                dz[z_values[i-1] <= 0] = 0
        
        return gradients_w, gradients_b
    
    def sgd_update(self, x, y, learning_rate=0.01):
        grad_w, grad_b = self.backward(x, y)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]
    
    def copy(self):
        """Create a deep copy of the model"""
        new_model = MLP(self.layers)
        new_model.set_params(self.get_params().copy())
        return new_model

class BioInspiredHybridOptimizer:
    """
    生物啟發的混合優化器：
    
    模擬自然演化：
    1. 每一代（Generation）生成多個個體（多尺度變異）
    2. 每個個體在其"生命週期"內學習（SGD優化）
    3. 選擇最佳個體的"基因"（參數）傳遞給下一代
    
    多尺度變異（Multi-Scale Mutation）：
    - 同一代中包含不同幅度的變異（小、中、大）
    - 自適應調整變異範圍
    """
    
    def __init__(self, model,
                 population_size=15,  # 較小族群，每個個體都會被優化
                 num_sigma_levels=3,   # 多尺度變異層級
                 initial_sigma=0.05,   # 初始變異範圍（較小）
                 max_sgd_lr=0.01,
                 min_sgd_lr=0.001,
                 sgd_steps_per_life=200,  # 每個個體的"生命長度"
                 batch_size=128):
        
        self.base_model = model
        self.pop_size = population_size
        self.num_sigma_levels = num_sigma_levels
        self.sigma = initial_sigma
        self.max_sgd_lr = max_sgd_lr
        self.min_sgd_lr = min_sgd_lr
        self.sgd_steps = sgd_steps_per_life
        self.batch_size = batch_size
        self.param_size = len(model.get_params())
        
        # Evolution path for CMA-like adaptation
        self.evolution_path = np.zeros(self.param_size)
        self.path_decay = 0.8
        
        print(f"\n生物啟發混合優化器配置:")
        print(f"  族群大小: {self.pop_size}")
        print(f"  變異層級: {self.num_sigma_levels} (小/中/大突變)")
        print(f"  每個體生命週期: {self.sgd_steps} SGD步驟")
        print(f"  初始變異範圍: {self.sigma}")
    
    def generate_multi_scale_population(self, parent_params):
        """
        生成多尺度變異的族群
        每一代包含不同範圍的突變，模擬自然界的多樣性
        """
        population = []
        sigma_used = []
        
        # 計算不同尺度的sigma
        sigma_levels = [
            self.sigma * 0.3,   # 小突變（微調）
            self.sigma * 1.0,   # 中等突變（探索）
            self.sigma * 3.0    # 大突變（跳出局部最優）
        ]
        
        individuals_per_level = self.pop_size // self.num_sigma_levels
        
        for level_idx, sigma_level in enumerate(sigma_levels):
            for _ in range(individuals_per_level):
                # 生成擾動
                noise = np.random.randn(self.param_size)
                offspring_params = parent_params + sigma_level * noise
                
                population.append(offspring_params)
                sigma_used.append(sigma_level)
        
        # 填充剩餘個體（如果pop_size不能被level數整除）
        while len(population) < self.pop_size:
            noise = np.random.randn(self.param_size)
            offspring_params = parent_params + self.sigma * noise
            population.append(offspring_params)
            sigma_used.append(self.sigma)
        
        return population, sigma_used
    
    def live_and_learn(self, individual_params, x_train, y_train, gen_progress):
        """
        個體的"生命週期"：在其基因基礎上通過經驗學習（SGD）
        
        類比：
        - individual_params: 遺傳的基因
        - SGD: 個體一生的學習和適應
        - 返回值: 個體達到的最佳狀態
        """
        # 創建個體模型
        individual = self.base_model.copy()
        individual.set_params(individual_params)
        
        # 學習率衰減（模擬從年輕到年老的學習能力）
        lr_schedule = lambda step: self.min_sgd_lr + \
            (self.max_sgd_lr - self.min_sgd_lr) * (1 - step / self.sgd_steps)
        
        # 在生命週期內學習
        best_fitness = 0
        best_params = individual_params.copy()
        
        for step in range(self.sgd_steps):
            # 隨機採樣訓練數據
            idx = np.random.choice(len(x_train), self.batch_size, replace=False)
            x_batch = x_train[idx]
            y_batch = y_train[idx]
            
            # 學習（SGD更新）
            current_lr = lr_schedule(step)
            individual.sgd_update(x_batch, y_batch, current_lr)
            
            # 定期評估（不是每步都評估，節省時間）
            if step % 50 == 0 or step == self.sgd_steps - 1:
                fitness = individual.evaluate(x_batch, y_batch)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = individual.get_params().copy()
        
        return best_params, best_fitness
    
    def update_evolution_path(self, selected_direction):
        """更新演化路徑（類似CMA-ES）"""
        self.evolution_path = self.path_decay * self.evolution_path + \
                              (1 - self.path_decay) * selected_direction
    
    def adapt_sigma(self, fitness_improvements):
        """
        根據族群的適應度改進情況調整變異範圍
        如果改進良好→減小sigma（利用）
        如果停滯→增大sigma（探索）
        """
        avg_improvement = np.mean(fitness_improvements)
        
        if avg_improvement > 0.02:  # 顯著進步
            self.sigma *= 0.95
        elif avg_improvement < 0.005:  # 停滯
            self.sigma *= 1.05
        
        # 限制範圍
        self.sigma = np.clip(self.sigma, 0.001, 0.3)
    
    def train(self, x_train, y_train, x_test, y_test, generations=50):
        """
        主訓練循環：模擬多代演化
        """
        history = {
            'train_acc': [], 'test_acc': [], 'train_loss': [],
            'best_individual_acc': [], 'time': [], 'samples_seen': [],
            'sigma': [], 'population_diversity': []
        }
        
        # 初始"祖先"
        best_parent_params = self.base_model.get_params()
        prev_best_fitness = 0
        
        total_samples = 0
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"開始演化訓練 ({generations} 代)")
        print(f"{'='*70}\n")
        
        for gen in range(generations):
            gen_start = time.time()
            gen_progress = gen / generations
            
            # 1. 生成多尺度變異的後代
            population, sigma_used = self.generate_multi_scale_population(best_parent_params)
            
            # 2. 每個個體"活著並學習"
            print(f"Generation {gen+1}/{generations} - 個體學習中...", end='', flush=True)
            
            fitness_list = []
            learned_population = []
            initial_fitness = []
            
            for idx, individual_params in enumerate(population):
                # 記錄初始適應度（遺傳的基因）
                temp_model = self.base_model.copy()
                temp_model.set_params(individual_params)
                eval_idx = np.random.choice(len(x_train), 500, replace=False)
                init_fit = temp_model.evaluate(x_train[eval_idx], y_train[eval_idx])
                initial_fitness.append(init_fit)
                
                # 個體學習
                learned_params, final_fitness = self.live_and_learn(
                    individual_params, x_train, y_train, gen_progress
                )
                
                learned_population.append(learned_params)
                fitness_list.append(final_fitness)
                total_samples += self.sgd_steps * self.batch_size
                
                if (idx + 1) % 5 == 0:
                    print(f".", end='', flush=True)
            
            print(" 完成")
            
            # 3. 選擇（自然選擇：最適者生存）
            best_idx = np.argmax(fitness_list)
            best_offspring_params = learned_population[best_idx]
            best_fitness = fitness_list[best_idx]
            best_individual_acc = fitness_list[best_idx]
            
            # 計算學習帶來的提升
            fitness_improvements = [fitness_list[i] - initial_fitness[i] 
                                   for i in range(len(fitness_list))]
            avg_improvement = np.mean(fitness_improvements)
            
            # 4. 更新演化路徑
            direction = best_offspring_params - best_parent_params
            self.update_evolution_path(direction)
            
            # 5. 自適應調整sigma
            self.adapt_sigma(fitness_improvements)
            
            # 6. 最佳個體成為下一代的"父母"
            best_parent_params = best_offspring_params
            self.base_model.set_params(best_parent_params)
            
            # 評估
            eval_idx = np.random.choice(len(x_train), 1000, replace=False)
            train_acc = self.base_model.evaluate(x_train[eval_idx], y_train[eval_idx])
            test_acc = self.base_model.evaluate(x_test, y_test)
            train_loss = self.base_model.compute_loss(x_train[eval_idx], y_train[eval_idx])
            
            # 計算族群多樣性
            population_std = np.std([np.linalg.norm(p - best_parent_params) 
                                    for p in learned_population])
            
            # 記錄
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['train_loss'].append(train_loss)
            history['best_individual_acc'].append(best_individual_acc)
            history['time'].append(time.time() - start_time)
            history['samples_seen'].append(total_samples)
            history['sigma'].append(self.sigma)
            history['population_diversity'].append(population_std)
            
            gen_time = time.time() - gen_start
            
            print(f"  最佳個體: {best_individual_acc:.4f} | "
                  f"測試準確率: {test_acc:.4f} | "
                  f"平均學習提升: {avg_improvement:.4f} | "
                  f"Sigma: {self.sigma:.4f} | "
                  f"代時間: {gen_time:.1f}s\n")
            
            prev_best_fitness = best_fitness
        
        return history

if __name__ == "__main__":
    print("="*70)
    print("生物啟發混合優化器：多尺度演化 + 個體學習")
    print("="*70)
    
    x_train, y_train, x_test, y_test = load_data()
    
    # 測試不同配置
    configs = {
        'Bio-Hybrid-Fast': {
            'pop_size': 10,
            'sgd_steps': 150,
            'generations': 50,
            'desc': '小族群，快速學習'
        },
        'Bio-Hybrid-Balanced': {
            'pop_size': 15,
            'sgd_steps': 200,
            'generations': 40,
            'desc': '平衡配置'
        },
        'Bio-Hybrid-Deep': {
            'pop_size': 20,
            'sgd_steps': 250,
            'generations': 30,
            'desc': '大族群，深度學習'
        }
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n{'='*70}")
        print(f"測試配置: {name}")
        print(f"說明: {config['desc']}")
        print(f"{'='*70}")
        
        model = MLP([784, 128, 64, 10])
        optimizer = BioInspiredHybridOptimizer(
            model=model,
            population_size=config['pop_size'],
            num_sigma_levels=3,
            initial_sigma=0.05,
            max_sgd_lr=0.01,
            min_sgd_lr=0.001,
            sgd_steps_per_life=config['sgd_steps'],
            batch_size=128
        )
        
        results[name] = optimizer.train(
            x_train, y_train, x_test, y_test,
            generations=config['generations']
        )
    
    # 可視化
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Test Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    for name, history in results.items():
        ax1.plot(history['test_acc'], label=name, linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy vs Generation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time Efficiency
    ax2 = fig.add_subplot(gs[0, 1])
    for name, history in results.items():
        ax2.plot(history['time'], history['test_acc'], label=name, linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Learning Curve (Time)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sample Efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    for name, history in results.items():
        samples_m = np.array(history['samples_seen']) / 1e6
        ax3.plot(samples_m, history['test_acc'], label=name, linewidth=2)
    ax3.set_xlabel('Samples Seen (Millions)')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Sample Efficiency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Sigma Evolution (Multi-scale adaptation)
    ax4 = fig.add_subplot(gs[1, 0])
    for name, history in results.items():
        ax4.plot(history['sigma'], label=name, linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Sigma (Mutation Range)')
    ax4.set_title('Adaptive Mutation Range')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Population Diversity
    ax5 = fig.add_subplot(gs[1, 1])
    for name, history in results.items():
        ax5.plot(history['population_diversity'], label=name, linewidth=2, alpha=0.7)
    ax5.set_xlabel('Generation')
    ax5.set_ylabel('Population Diversity (Std of Params)')
    ax5.set_title('Population Diversity Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Training Loss
    ax6 = fig.add_subplot(gs[1, 2])
    for name, history in results.items():
        ax6.plot(history['train_loss'], label=name, linewidth=2)
    ax6.set_xlabel('Generation')
    ax6.set_ylabel('Training Loss')
    ax6.set_title('Training Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # Learning Impact (Best Individual vs Final)
    ax7 = fig.add_subplot(gs[2, 0])
    for name, history in results.items():
        improvements = [history['test_acc'][i] - history['best_individual_acc'][i] 
                       for i in range(len(history['test_acc']))]
        ax7.plot(improvements, label=name, linewidth=2, alpha=0.7)
    ax7.set_xlabel('Generation')
    ax7.set_ylabel('Learning Improvement')
    ax7.set_title('Within-Life Learning Impact')
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Convergence Comparison
    ax8 = fig.add_subplot(gs[2, 1:])
    methods = list(results.keys())
    final_acc = [results[m]['test_acc'][-1] for m in methods]
    best_acc = [max(results[m]['test_acc']) for m in methods]
    final_time = [results[m]['time'][-1] for m in methods]
    total_samples = [results[m]['samples_seen'][-1] / 1e6 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.2
    
    ax8.bar(x - 1.5*width, final_acc, width, label='Final Test Acc', alpha=0.8)
    ax8.bar(x - 0.5*width, best_acc, width, label='Best Test Acc', alpha=0.8)
    ax8.bar(x + 0.5*width, [t/max(final_time) for t in final_time], width,
           label='Relative Time', alpha=0.8)
    ax8.bar(x + 1.5*width, [s/max(total_samples) for s in total_samples], width,
           label='Relative Samples', alpha=0.8)
    
    ax8.set_ylabel('Value')
    ax8.set_title('Final Performance Comparison')
    ax8.set_xticks(x)
    ax8.set_xticklabels(methods, rotation=15, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('bio_inspired_hybrid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 結果摘要
    print("\n" + "="*70)
    print("最終結果摘要")
    print("="*70)
    for name, history in results.items():
        print(f"\n{name}:")
        print(f"  最終測試準確率: {history['test_acc'][-1]:.4f}")
        print(f"  最佳測試準確率: {max(history['test_acc']):.4f}")
        print(f"  總訓練時間: {history['time'][-1]:.1f}秒")
        print(f"  總樣本數: {history['samples_seen'][-1]/1e6:.2f}M")
        print(f"  最終Sigma: {history['sigma'][-1]:.6f}")
        print(f"  樣本效率: {history['test_acc'][-1]/(history['samples_seen'][-1]/1e6):.4f} acc/M")
