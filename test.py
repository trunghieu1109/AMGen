import matplotlib.pyplot as plt
import numpy as np

# Các trục (tiêu chí đánh giá)
labels = ['Accuracy', 'Speed', 'Cost', 'Stability', 'Scalability']
num_vars = len(labels)

# Giá trị mẫu
values = [0.8, 0.6, 0.3, 0.9, 0.7]
# Đóng vòng tròn bằng cách lặp lại phần tử đầu
values += values[:1]

# Góc cho từng trục
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Tạo biểu đồ
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values, color='blue', linewidth=2)
ax.fill(angles, values, color='blue', alpha=0.25)

# Thêm nhãn cho từng trục
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('Model Performance (Spider Plot)')
plt.savefig("spider_plot.jpg")
