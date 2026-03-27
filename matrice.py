import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

classes = [
    'catina', 'galbenele', 'ienupar', 'lavanda', 'lumanarica',
    'maces', 'menta', 'musetel', 'papadie', 'papalau',
    'pelin', 'podbal', 'salcie', 'scaivanat', 'soc',
    'spanz', 'sunatoare', 'treifratipatati', 'volbura'
]

cm = np.zeros((19, 19), dtype=int)
cm[0, 0] = 12
cm[1, 1] = 11
cm[2, 2] = 9; cm[2, 14] = 1
cm[3, 3] = 17; cm[3, 4] = 1; cm[3, 10] = 1; cm[3, 15] = 1; cm[3, 16] = 2
cm[4, 4] = 9; cm[4, 12] = 1; cm[4, 13] = 1; cm[4, 14] = 2; cm[4, 15] = 1
cm[5, 0] = 1; cm[5, 5] = 12; cm[5, 9] = 2; cm[5, 14] = 1; cm[5, 16] = 1
cm[6, 4] = 1; cm[6, 6] = 12; cm[6, 14] = 2
cm[7, 0] = 1; cm[7, 3] = 1; cm[7, 7] = 13
cm[8, 3] = 1; cm[8, 7] = 1; cm[8, 8] = 7
cm[9, 0] = 1; cm[9, 9] = 12
cm[10, 2] = 1; cm[10, 3] = 1; cm[10, 10] = 7; cm[10, 13] = 1; cm[10, 14] = 3
cm[11, 11] = 11; cm[11, 15] = 1
cm[12, 3] = 2; cm[12, 10] = 1; cm[12, 12] = 5
cm[13, 13] = 13
cm[14, 6] = 1; cm[14, 14] = 13; cm[14, 15] = 3
cm[15, 12] = 1; cm[15, 14] = 1; cm[15, 15] = 24
cm[16, 7] = 1; cm[16, 10] = 2; cm[16, 16] = 11
cm[17, 2] = 1; cm[17, 4] = 1; cm[17, 7] = 1; cm[17, 14] = 1; cm[17, 17] = 9
cm[18, 7] = 1; cm[18, 10] = 1; cm[18, 14] = 1; cm[18, 15] = 4; cm[18, 18] = 10

plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
plt.title('Matricea de Confuzie - Analiza Botanica AI')
plt.xlabel('Predictie Model')
plt.ylabel('Adevarat (Ground Truth)')
plt.savefig('matrice_confuzie_finala.png')
plt.show()

y_true, y_pred = [], []
for i in range(19):
    for j in range(19):
        count = cm[i, j]
        if count > 0:
            y_true.extend([classes[i]] * count)
            y_pred.extend([classes[j]] * count)

report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('metrici_performanta.csv')

print("=== REZUMAT PERFORMANTA ===")
print(f"Acuratete Generala: {report_df.loc['accuracy', 'precision']:.2%}")
print(f"Precision Mediu: {report_df.loc['macro avg', 'precision']:.2%}")
print(f"Recall (Senzitivitate) Mediu: {report_df.loc['macro avg', 'recall']:.2%}")
print(f"Scor F1 Mediu: {report_df.loc['macro avg', 'f1-score']:.2%}")