import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math

models = ['gpt-4.1-mini', 'gpt-4o', 'o4-mini']
cost = [0.09, 0.258, 0.20]
accuracy = [83.33, 66.67, 89.58]
time = [387.71, 342.49, 411.68]

# Gán mỗi model một màu riêng
colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red

plt.figure(figsize=(10, 6))
bubble_sizes = [c * 40000 for c in cost]

scatter = plt.scatter(
    time, accuracy,
    s=bubble_sizes,
    c=colors,
    alpha=0.7
)

# Annotate và thêm tâm và đường nét đứt
for i, model in enumerate(models):
    # Vẽ tâm của bubble
    plt.plot(time[i], accuracy[i], '+', markersize=4)  # 'ko' = black circle

    # Vẽ đường nét đứt đến trục X
    plt.plot([time[i], time[i]], [0, accuracy[i]], linestyle='dotted', color='gray', alpha=0.6)

    # Vẽ đường nét đứt đến trục Y
    plt.plot([0, time[i]], [accuracy[i], accuracy[i]], linestyle='dotted', color='gray', alpha=0.6)

    # Hiển thị tên model và cost ở phía trên bubble
    text_offset = - (math.sqrt((bubble_sizes[i]) / math.pi) + 20)
    plt.annotate(f"{model}\n(${cost[i]})",
                 (time[i], accuracy[i]),
                 xytext=(0, text_offset),
                 textcoords='offset points',
                 ha='center')

# Thiết lập trục và biểu đồ
plt.xlabel('Time (s)')
plt.ylabel('Accuracy (%)')
plt.xlim(250, 450)
plt.ylim(40, 100)
plt.title('Execution Model Analysis: Accuracy vs Time vs Cost')
plt.grid(linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("execution_model_analysis.png")
plt.show()
