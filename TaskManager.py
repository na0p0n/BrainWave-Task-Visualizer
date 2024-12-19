import random

class TaskManager:
    def __init__(self, count):
        self.counters = {
            "right": 0,
            "left": 0,
            "neutral": 0
        }
        self.taskCounters = {
            "screen": 0,
            "sound": 0,
            "task": 0
        }
        self.target_count = count  # 各タイプを実行する目標回数

    def get_next_type(self, mode):
        if mode == "mind":
            counters = self.counters
            available_types = ["right", "left", "neutral"]
        elif mode == "task":
            counters = self.taskCounters
            available_types = ["screen", "sound", "task"]
        
        # すべてのタスクが目標回数に達しているか確認
        if all(count >= self.target_count for count in counters.values()):
            return None  # すべて完了
            
        # 目標回数に達していないタスクの中からランダムに選択
        available_tasks = [
            task_type for task_type in available_types 
            if counters[task_type] < self.target_count
        ]
        if not available_tasks:
            return None
        
        weights = [1 / ((counters[task] + 1) ** 2) for task in available_tasks]
        selected_task = random.choices(available_tasks, weights=weights, k=1)[0]
        counters[selected_task] += 1
        return selected_task

    def get_counts(self):
        return self.counters
    
    def get_taskCounts(self):
        return self.taskCounters
    
    def sub_counts(self, type, task, amount):
        self.counters[type] -= amount
        self.taskCounters[task] -= amount