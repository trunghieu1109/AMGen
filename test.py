import matplotlib.pyplot as plt
import math

models = ['gpt-4.1-mini', 'gpt-4o', 'o4-mini']
cost = [0.09, 0.258, 0.20]
accuracy = [83.33, 66.67, 89.58]
time = [387.71, 342.49, 411.68]
colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red

# Điều chỉnh diện tích bubble theo cost
scale_factor = 35000  # <-- thay đổi giá trị này để tăng/giảm kích thước bubble
bubble_areas = [c * scale_factor for c in cost]

plt.figure(figsize=(8, 6))

# Vẽ scatter bubble
scatter = plt.scatter(
    time, accuracy,
    s=bubble_areas,  # diện tích pixel^2
    c=colors,
    alpha=0.5,
    edgecolors='black',
    linewidths=1.2,
    zorder=2
)

# Vẽ tâm và đường nét đứt
for i in range(len(models)):
    # Vẽ tâm
    plt.plot(time[i], accuracy[i], marker='+', color='black', markersize=8, zorder=3)

    # Đường nét đứt
    plt.plot([time[i], time[i]], [40, accuracy[i]], linestyle='dotted', color='gray', alpha=0.6)
    plt.plot([250, time[i]], [accuracy[i], accuracy[i]], linestyle='dotted', color='gray', alpha=0.6)

    # Ghi nhãn
    offset = -math.sqrt(bubble_areas[i] / math.pi) - 20
    plt.annotate(
        f"{models[i]}\n(${cost[i]})",
        (time[i], accuracy[i]),
        xytext=(0, offset),
        textcoords='offset points',
        ha='center',
        fontsize=9
    )

# Thiết lập trục
plt.xlim(300, 450)
plt.ylim(50, 100)
plt.xlabel("Time (s)")
plt.ylabel("Accuracy (%)")
plt.title("Execution Model Analysis: Accuracy vs Time vs Cost", fontsize=12, weight='bold')
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("execution_model_scatter.png", dpi=300)
plt.show()
