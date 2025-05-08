
y_scores = knn.predict_proba(X_test)[:, 1]

precision_points = []
recall_points = []

# 遍历不同的阈值
thresholds = np.linspace(0, 1, 100)  # 生成100个阈值
for threshold in thresholds:
    y_pred = (y_scores >= threshold).astype(int)
    # print("y_pred:",y_pred)
    TP = np.sum((y_pred == 1) & (y_test == 1))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))
    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    precision_points.append(precision)
    recall_points.append(recall)

print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)
print("R:", recall_points)
print("P:", precision_points)

# 绘制P-R曲线
plt.plot(recall_points, precision_points, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P-R Curve')
plt.grid(True)
plt.show()