# =============================================================================
# REGULARIZED LOGISTIC REGRESSION EXPERIMENTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from scipy import stats
from scipy.sparse import csr_matrix

from sklearn.datasets import (
    make_classification,
    load_breast_cancer,
    load_digits
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures
)

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)

from sklearn.dummy import DummyClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
N_JOBS = -1

# =============================================================================
# 1. SYNTHETIC HIGH-DIMENSIONAL DATASET
# =============================================================================

print("=" * 80)
print("EXPERIMENT 1: SYNTHETIC HIGH-DIMENSIONAL DATA (p >> n)")
print("=" * 80)

X_syn, y_syn = make_classification(
    n_samples=500,
    n_features=2000,
    n_informative=50,
    n_redundant=100,
    n_classes=2,
    flip_y=0.05,
    random_state=RANDOM_STATE
)

print(f"Synthetic dataset: {X_syn.shape}")

# Proper stratified split
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_syn,
    y_syn,
    test_size=TEST_SIZE,
    stratify=y_syn,
    random_state=RANDOM_STATE
)

syn_scaler = StandardScaler() # Renamed to avoid confusion with X_scaled later

X_train_syn_s = syn_scaler.fit_transform(X_train_syn)
X_test_syn_s = syn_scaler.transform(X_test_syn)

Cs_syn = np.logspace(-4, 2, 15)
l1_ratios = [0.2, 0.4, 0.6, 0.8]

records_syn = []

# =============================================================================
# L2 REGULARIZATION
# =============================================================================

print("\nTraining L2 models...")

for C in Cs_syn:

    model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=C,
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(X_train_syn_s, y_train_syn)

    y_pred = model.predict(X_test_syn_s)

    coef = model.coef_.ravel()

    records_syn.append({
        'reg_type': 'L2',
        'C': C,
        'accuracy': accuracy_score(y_test_syn, y_pred),
        'f1': f1_score(y_test_syn, y_pred),
        'sparsity': np.sum(np.abs(coef) > 1e-6)
    })

# =============================================================================
# L1 REGULARIZATION
# =============================================================================

print("Training L1 models...")

for C in Cs_syn:

    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=C,
        max_iter=1000,
        random_state=RANDOM_STATE,
        tol=1e-3
    )

    model.fit(X_train_syn_s, y_train_syn)

    y_pred = model.predict(X_test_syn_s)

    coef = model.coef_.ravel()

    records_syn.append({
        'reg_type': 'L1',
        'C': C,
        'accuracy': accuracy_score(y_test_syn, y_pred),
        'f1': f1_score(y_test_syn, y_pred),
        'sparsity': np.sum(np.abs(coef) > 1e-6)
    })

# =============================================================================
# ELASTIC NET
# =============================================================================

print("Training ElasticNet models...")

for l1_ratio in l1_ratios:

    for C in Cs_syn:

        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=C,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=RANDOM_STATE,
            tol=1e-3
        )

        model.fit(X_train_syn_s, y_train_syn)

        y_pred = model.predict(X_test_syn_s)

        coef = model.coef_.ravel()

        records_syn.append({
            'reg_type': f'ElasticNet_{l1_ratio}',
            'C': C,
            'accuracy': accuracy_score(y_test_syn, y_pred),
            'f1': f1_score(y_test_syn, y_pred),
            'sparsity': np.sum(np.abs(coef) > 1e-6)
        })

df_syn = pd.DataFrame(records_syn)

# =============================================================================
# SYNTHETIC PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]

for reg in df_syn['reg_type'].unique():

    sub = df_syn[df_syn['reg_type'] == reg]

    ax1.semilogx(
        sub['C'],
        sub['accuracy'],
        'o-',
        label=reg
    )

ax1.set_title('Synthetic Accuracy')
ax1.set_xlabel('C')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2 = axes[1]

for reg in df_syn['reg_type'].unique():

    if reg == 'L2':
        continue

    sub = df_syn[df_syn['reg_type'] == reg]

    ax2.semilogx(
        sub['C'],
        sub['sparsity'],
        'o-',
        label=reg
    )

ax2.set_title('Sparsity')
ax2.set_xlabel('C')
ax2.set_ylabel('Non-zero Coefficients')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('synthetic_highdim_results.png', dpi=300)

print("✓ Saved synthetic_highdim_results.png")

# =============================================================================
# 2. BREAST CANCER + POLYNOMIAL FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: BREAST CANCER POLYNOMIAL")
print("=" * 80)

X_bc, y_bc = load_breast_cancer(return_X_y=True)

poly2 = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly2 = csr_matrix(poly2.fit_transform(X_bc))

print(f"Polynomial features shape: {X_poly2.shape}")

X_train_p2, X_test_p2, y_train_p2, y_test_p2 = train_test_split(
    X_poly2,
    y_bc,
    test_size=TEST_SIZE,
    stratify=y_bc,
    random_state=RANDOM_STATE
)

scaler_p2 = StandardScaler(with_mean=False)

X_train_p2_s = scaler_p2.fit_transform(X_train_p2)
X_test_p2_s = scaler_p2.transform(X_test_p2)

Cs_p2 = np.logspace(-4, 2, 20)

records_p2 = []

# L2
for C in Cs_p2:

    model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=C,
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(X_train_p2_s, y_train_p2)

    y_pred = model.predict(X_test_p2_s)

    coef = model.coef_.ravel()

    records_p2.append({
        'reg_type': 'L2',
        'C': C,
        'test_acc': accuracy_score(y_test_p2, y_pred),
        'sparsity': np.sum(np.abs(coef) > 1e-6)
    })

# ElasticNet
for l1_ratio in [0.3, 0.5, 0.7, 0.9]:

    for C in Cs_p2:

        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=C,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=RANDOM_STATE
        )

        model.fit(X_train_p2_s, y_train_p2)

        y_pred = model.predict(X_test_p2_s)

        coef = model.coef_.ravel()

        records_p2.append({
            'reg_type': f'EN_{l1_ratio}',
            'C': C,
            'test_acc': accuracy_score(y_test_p2, y_pred),
            'sparsity': np.sum(np.abs(coef) > 1e-6)
        })

df_p2 = pd.DataFrame(records_p2)

# =============================================================================
# POLYNOMIAL PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]

for reg in df_p2['reg_type'].unique():

    sub = df_p2[df_p2['reg_type'] == reg]

    ax1.semilogx(
        sub['C'],
        sub['test_acc'],
        'o-',
        label=reg
    )

ax1.set_title('Polynomial Accuracy')
ax1.legend()
ax1.grid(True)

ax2 = axes[1]

for reg in df_p2['reg_type'].unique():

    if reg == 'L2':
        continue

    sub = df_p2[df_p2['reg_type'] == reg]

    ax2.semilogx(
        sub['C'],
        sub['sparsity'],
        'o-',
        label=reg
    )

ax2.set_title('Polynomial Sparsity')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('poly_degree3_results.png', dpi=300)

print("✓ Saved poly_degree3_results.png")

# =============================================================================
# 3. DIGITS DATASET GRID SEARCH
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 3: DIGITS")
print("=" * 80)

X_digits, y_digits = load_digits(return_X_y=True)

y_binary = (y_digits == 3).astype(int)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits,
    y_binary,
    test_size=TEST_SIZE,
    stratify=y_binary,
    random_state=RANDOM_STATE
)

# =============================================================================
# L2 PIPELINE
# =============================================================================

l2_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

l2_grid = GridSearchCV(
    l2_pipeline,
    {'clf__C': np.logspace(-4, 2, 20)},
    cv=CV_FOLDS,
    scoring='accuracy',
    n_jobs=N_JOBS
)

l2_grid.fit(X_train_d, y_train_d)

# =============================================================================
# L1 PIPELINE
# =============================================================================

l1_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

l1_grid = GridSearchCV(
    l1_pipeline,
    {'clf__C': np.logspace(-4, 2, 20)},
    cv=CV_FOLDS,
    scoring='accuracy',
    n_jobs=N_JOBS
)

l1_grid.fit(X_train_d, y_train_d)

# =============================================================================
# ELASTICNET PIPELINE
# =============================================================================

en_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

en_grid = GridSearchCV(
    en_pipeline,
    {
        'clf__C': np.logspace(-4, 2, 15),
        'clf__l1_ratio': [0.3, 0.5, 0.7, 0.9]
    },
    cv=CV_FOLDS,
    scoring='accuracy',
    n_jobs=N_JOBS
)

en_grid.fit(X_train_d, y_train_d)

best_l2 = l2_grid.best_estimator_
best_l1 = l1_grid.best_estimator_
best_en = en_grid.best_estimator_

print("\nBest Parameters")

print(l2_grid.best_params_)
print(l1_grid.best_params_)
print(en_grid.best_params_)

# =============================================================================
# DUMMY CLASSIFIER
# =============================================================================

dummy = DummyClassifier(
    strategy='most_frequent',
    random_state=RANDOM_STATE
)

dummy.fit(X_train_d, y_train_d)

dummy_pred = dummy.predict(X_test_d)

print("\nDummy Baseline")
print("Accuracy:", accuracy_score(y_test_d, dummy_pred))
print("F1:", f1_score(y_test_d, dummy_pred))

# =============================================================================
# TEST PERFORMANCE
# =============================================================================

for name, model in [
    ('L2', best_l2),
    ('L1', best_l1),
    ('ElasticNet', best_en)
]:

    y_pred = model.predict(X_test_d)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test_d, y_pred))
    print("F1:", f1_score(y_test_d, y_pred))

# =============================================================================
# 4. CROSS VALIDATION ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("CROSS VALIDATION ANALYSIS")
print("=" * 80)

X_bc_full, y_bc_full = load_breast_cancer(return_X_y=True)

cv10 = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)

model_l2 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=0.0281,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

model_l1 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l1',
        solver='saga',
        C=3.7276,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

model_en = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=0.0869,
        l1_ratio=0.5,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

scores_l2 = cross_val_score(
    model_l2,
    X_bc_full,
    y_bc_full,
    cv=cv10,
    scoring='accuracy'
)

scores_l1 = cross_val_score(
    model_l1,
    X_bc_full,
    y_bc_full,
    cv=cv10,
    scoring='accuracy'
)

scores_en = cross_val_score(
    model_en,
    X_bc_full,
    y_bc_full,
    cv=cv10,
    scoring='accuracy'
)

f_stat, p_value = stats.f_oneway(
    scores_l2,
    scores_l1,
    scores_en
)

print(f"ANOVA p-value: {p_value:.5f}")

# =============================================================================
# 5. ROC CURVE
# =============================================================================

print("\nGenerating ROC Curves...")

X_train_roc, X_test_roc, y_train_roc, y_test_roc = train_test_split(
    X_bc_full,
    y_bc_full,
    test_size=TEST_SIZE,
    stratify=y_bc_full,
    random_state=RANDOM_STATE
)

plt.figure(figsize=(8, 6))

for name, model in [
    ('L2', model_l2),
    ('L1', model_l1),
    ('ElasticNet', model_en)
]:

    model.fit(X_train_roc, y_train_roc)

    probs = model.predict_proba(X_test_roc)[:, 1]

    fpr, tpr, _ = roc_curve(y_test_roc, probs)

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f'{name} (AUC={roc_auc:.3f})'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)

print("✓ Saved roc_curves.png")

# =============================================================================
# 6. LEARNING CURVE
# =============================================================================

print("\nGenerating improved learning curve...")

learning_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=0.0281,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

train_sizes, train_scores, test_scores = learning_curve(
    learning_pipeline,
    X_bc_full,
    y_bc_full,
    cv=cv10,
    n_jobs=N_JOBS,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))

# Confidence intervals
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
    label='Training Std'
)

plt.fill_between(
    train_sizes,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.15,
    label='Validation Std'
)

# Curves
plt.plot(
    train_sizes,
    train_mean,
    'o-',
    linewidth=2,
    markersize=8,
    label='Training Accuracy'
)

plt.plot(
    train_sizes,
    test_mean,
    's-',
    linewidth=2,
    markersize=8,
    label='Cross-Validation Accuracy'
)

# Annotation
plt.annotate(
    f'{train_mean[-1]:.3f}',
    (train_sizes[-1], train_mean[-1]),
    textcoords="offset points",
    xytext=(10, 5)
)

plt.annotate(
    f'{test_mean[-1]:.3f}',
    (train_sizes[-1], test_mean[-1]),
    textcoords="offset points",
    xytext=(10, -15)
)

plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')

plt.title('Improved Learning Curve: L2 Logistic Regression')

plt.legend(loc='lower right')

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('learning_curve_improved_v2.png', dpi=300)

print("✓ Saved: learning_curve_improved_v2.png")

# =============================================================================
# 9. FEATURE IMPORTANCE HEATMAP
# =============================================================================

print("\nGenerating improved feature heatmap...")

X_bc_scaled = StandardScaler().fit_transform(X_bc_full)

l1_model_full = LogisticRegression(
    penalty='l1',
    solver='saga',
    C=3.7276,
    max_iter=1000,
    random_state=RANDOM_STATE
)

l1_model_full.fit(X_bc_scaled, y_bc_full)

coef = l1_model_full.coef_.ravel()

coef_abs = np.abs(coef)

feature_names = load_breast_cancer()['feature_names']

# Top features
top20_idx = np.argsort(coef_abs)[-20:][::-1]

top_features = feature_names[top20_idx]
top_values = coef[top20_idx]

plt.figure(figsize=(10, 7))
colors = ['coral' if val < 0 else 'royalblue' for val in top_values]
bars = plt.barh(top_features, top_values, color=colors, edgecolor='black', linewidth=0.5)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Most Important Features (L1 Regularization)', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)

# Add value labels on the bars
for bar, val in zip(bars, top_values):
    plt.text(val + (0.01 if val > 0 else -0.01),
             bar.get_y() + bar.get_height()/2,
             f'{val:.3f}',
             va='center',
             ha='left' if val > 0 else 'right',
             fontsize=8)

plt.tight_layout()
plt.savefig('feature_importance_improved.png', dpi=300)

print("✓ Saved: feature_importance_improved.png")

# =============================================================================
# 8. SUMMARY TABLE
# =============================================================================

summary_data = []

for reg in df_syn['reg_type'].unique():

    sub = df_syn[df_syn['reg_type'] == reg]

    best = sub.loc[sub['accuracy'].idxmax()]

    summary_data.append({
        'Dataset': 'Synthetic',
        'Regularizer': reg,
        'Best C': best['C'],
        'Accuracy': best['accuracy'],
        'Sparsity': best['sparsity']
    })

summary_data.append({
    'Dataset': 'Digits',
    'Regularizer': 'L1',
    'Best C': l1_grid.best_params_['clf__C'],
    'Accuracy': accuracy_score(
        y_test_d,
        best_l1.predict(X_test_d)
    ),
    'Sparsity': np.sum(
        np.abs(
            best_l1.named_steps['clf'].coef_.ravel()
        )
    ) > 1e-6
})

summary_data.append({
    'Dataset': 'Digits',
    'Regularizer': 'ElasticNet',
    'Best C': en_grid.best_params_['clf__C'],
    'Accuracy': accuracy_score(
        y_test_d,
        best_en.predict(X_test_d)
    ),
    'Sparsity': np.sum(
        np.abs(
            best_en.named_steps['clf'].coef_.ravel()
        )
    ) > 1e-6
})

df_summary = pd.DataFrame(summary_data)

print("\nFinal Summary")
print(df_summary)

df_summary.to_csv(
    'all_experiments_summary.csv',
    index=False
)

print("\n✓ Saved all_experiments_summary.csv")

# =============================================================================
# FINISHED
# =============================================================================

print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
print("=" * 80)

print("""
Generated Files:

1. synthetic_highdim_results.png
2. poly_degree3_results.png
3. roc_curves.png
4. learning_curve_improved_v2.png
5. feature_importance_improved.png
6. all_experiments_summary.csv
""")