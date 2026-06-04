import sqlite3
import matplotlib.pyplot as plt

db = sqlite3.connect("/root/autodl-tmp/ai-toolkit/output/Flux2_lora_v5/loss_log.db")
rows = db.execute("""
  SELECT s.step, m.value_real
  FROM steps s JOIN metrics m ON s.step = m.step
  WHERE m.key = 'loss/loss'
  ORDER BY s.step
""").fetchall()
db.close()

steps, losses = zip(*rows)

# 原始 loss + 滑动平均
window = 100
smoothed = [sum(losses[max(0,i-window):i+1])/len(losses[max(0,i-window):i+1]) for i in range(len(losses))]

plt.figure(figsize=(12, 4))
plt.plot(steps, losses, alpha=0.2, label='raw loss')
plt.plot(steps, smoothed, color='red', label=f'moving avg ({window})')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_curve_v5.png', dpi=150)
print("saved to loss_curve_v5.png")