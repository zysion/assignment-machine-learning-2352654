import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss
from data import read_data

X_train, X_val, y_train, y_val = read_data()

y_train = y_train.values.ravel()
y_val = y_val.values.ravel()

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)

print(f"Accuracy on Validation: {accuracy_score(y_val, y_pred):.4f}")

loss = log_loss(y_val, y_prob)
print(f"Log Loss (Calibration diagnostic): {loss:.4f}")

# Filter fail samples
errors_indices = np.where(y_pred != y_val)[0]

if len(errors_indices) > 0:
    # fail predicted but with high confidence
    predicted_probs = np.max(y_prob[errors_indices], axis=1) 
    worst_idx_relative = np.argmax(predicted_probs)
    worst_idx = errors_indices[worst_idx_relative]
    
    print("-" * 60)
    print(f"Index fail sample (in validation): {worst_idx}")
    print(f"True Label: {y_val[worst_idx]}")
    print(f"Predicted: {y_pred[worst_idx]}")
    print(f"Predicted probability for label {y_pred[worst_idx]}: {np.max(y_prob[worst_idx]):.4f}")
    
    # print("\nGiải thích cho MODEL_LOG:")
    # print("Mô hình đưa ra dự đoán sai lệch hoàn toàn so với thực tế nhưng lại có độ tự tin (xác suất) cực kỳ cao. "
    #       "Điều này gợi ý rằng giả định 'các đặc trưng độc lập có điều kiện' của thuật toán Naive Bayes "
    #       "đã bị vi phạm nghiêm trọng, do dữ liệu từ cảm biến chuyển động (gia tốc kế, con quay hồi chuyển) "
    #       "thực chất có độ tương quan rất chặt chẽ với nhau.")