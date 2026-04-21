import pandas as pd
import kagglehub
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score,
                             precision_score, recall_score, make_scorer)
from sklearn.preprocessing import StandardScaler, MinMaxScaler


path = kagglehub.dataset_download("ritwikb3/heart-disease-cleveland")
csv_path = os.path.join(path, 'Heart_disease_cleveland_new.csv')
df = pd.read_csv(csv_path)

print(df.head(10))
print(df.isna().sum())
print(df.dtypes)

X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(f"Ilość danych do nauki: {len(X_train)}")
#print(f"Ilość danych do testu: {len(X_test)}")

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Macierz pomyłek")
plt.savefig("confusion_matrix.png")


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Losowy klasyfikator")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Krzywa ROC")
plt.legend()
plt.savefig("roc_curve.png")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
#print(f"Precyzja: {precision:.2f}")
#print(f"Czułość (Recall): {recall:.2f}")


scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

model_std = LogisticRegression(max_iter=1000)
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)
y_prob_std = model_std.predict_proba(X_test_std)[:, 1]


scaler_norm = MinMaxScaler()
X_train_norm = scaler_norm.fit_transform(X_train)
X_test_norm = scaler_norm.transform(X_test)

model_norm = LogisticRegression(max_iter=1000)
model_norm.fit(X_train_norm, y_train)
y_pred_norm = model_norm.predict(X_test_norm)
y_prob_norm = model_norm.predict_proba(X_test_norm)[:, 1]

for name, y_p, y_pr in [
    ("Bez skalowania",  y_pred,      y_prob),
    ("Standaryzacja",   y_pred_std,  y_prob_std),
    ("Normalizacja",    y_pred_norm, y_prob_norm),
]:
    print(f"\n{name}:")
    print(f"  Precyzja: {precision_score(y_test, y_p):.2f}")  
    print(f"  Czułość:  {recall_score(y_test, y_p):.2f}")     
    print(f"  AUC:      {roc_auc_score(y_test, y_pr):.2f}") 

for name, model_x in [
    ("Bez skalowania", model),
    ("Standaryzacja",  model_std),
    ("Normalizacja",   model_norm),
]:
    print(f"{name}: {model_x.n_iter_[0]} iteracji")


C_values = [0.001, 0.01, 0.1, 1, 10, 100]
l1_ratios = [0, 0.5, 1]

print(f"\n{'C':<8} {'l1_ratio':<12} {'Precyzja':<12} {'Czułość':<12} {'AUC':<8} {'Iteracje'}")
print("-" * 60)

for C in C_values:
    for l1_ratio in l1_ratios:
        m = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=C,
            l1_ratio=l1_ratio,
            max_iter=10000
        )
        m.fit(X_train_std, y_train)  
        y_p = m.predict(X_test_std)
        y_pr = m.predict_proba(X_test_std)[:, 1]

        print(f"{C:<8} {l1_ratio:<12} "
              f"{precision_score(y_test, y_p, zero_division=0):<12.2f} "
              f"{recall_score(y_test, y_p):<12.2f} "
              f"{roc_auc_score(y_test, y_pr):<8.2f} "
              f"{m.n_iter_[0]}")
        

scoring = {
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score),
    'roc_auc': 'roc_auc'
}

model_cv = LogisticRegression(penalty='elasticnet', solver='saga',
                               C=0.1, l1_ratio=1, max_iter=10000)

print("-" * 45)

for k in [2, 5, 10]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for name, cv in [("zwykły", kf), ("stratyfikowany", skf)]:
        cv_results = cross_validate(model_cv, X_train_std, y_train,
                                    cv=cv, scoring=scoring)
        
        prec = cv_results['test_precision']
        rec = cv_results['test_recall']
        auc = cv_results['test_roc_auc']
        print(f"\n{k}-fold {name}:")
        print(f"  Precyzja: {prec.mean():.2f} +/- {prec.std():.2f}")
        print(f"  Czułość:  {rec.mean():.2f} +/- {rec.std():.2f}")
        print(f"  AUC:      {auc.mean():.2f} +/- {auc.std():.2f}")


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
metrics = ['test_precision', 'test_recall', 'test_roc_auc']
titles = ['Precyzja', 'Czułość', 'AUC']

for ax, metric, title in zip(axes, metrics, titles):
    data = []
    labels = []
    for k in [2, 5, 10]:
        for name, cv in [("zwykły", KFold(n_splits=k, shuffle=True, random_state=42)),
                         ("strat.", StratifiedKFold(n_splits=k, shuffle=True, random_state=42))]:
            res = cross_validate(model_cv, X_train_std, y_train, cv=cv, scoring=scoring)
            data.append(res[metric])
            labels.append(f"{k}-fold\n{name}")
    
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(title)
    ax.tick_params(axis='x', labelsize=7)

plt.tight_layout()
plt.savefig("cross_validation.png")
plt.show()