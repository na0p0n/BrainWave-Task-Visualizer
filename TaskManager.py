import random

class TaskManager:
    def __init__(self, count):
        self.counters = {
            "right": 0,
            "left": 0,
            "neutral": 0
        }
        self.target_count = 5  # 各タイプを実行する目標回数

    def get_next_task(self):
        # すべてのタスクが目標回数に達しているか確認
        if all(count >= self.target_count for count in self.counters.values()):
            return None  # すべて完了
            
        # 目標回数に達していないタスクの中からランダムに選択
        available_tasks = [
            task for task, count in self.counters.items() 
            if count < self.target_count
        ]
        if not available_tasks:
            return None
        selected_task = random.choice(available_tasks)
        self.counters[selected_task] += 1
        return selected_task

    def get_counts(self):
        return self.counters
    
    def sub_counts(self, task, amount):
        self.counters[task] -= amount