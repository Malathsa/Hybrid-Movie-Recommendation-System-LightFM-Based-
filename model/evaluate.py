from lightfm.evaluation import precision_at_k, auc_score, recall_at_k

def evaluate_model(model, train_interactions, test_interactions, item_features, k=10):
    # On training data
    prec_tr = precision_at_k(model, train_interactions, item_features=item_features, 
                             k=k, num_threads=1).mean()
    auc_tr = auc_score(model, train_interactions, item_features=item_features, 
                       num_threads=1).mean()
    rec_tr = recall_at_k(model, train_interactions, item_features=item_features,
                         k=k, num_threads=1).mean()   # ⬅️ (سطر جديد) Recall@k على التدريب

    # On test data
    prec_te = precision_at_k(model, test_interactions, train_interactions=train_interactions,
                             item_features=item_features, k=k, num_threads=1).mean()
    auc_te = auc_score(model, test_interactions, train_interactions=train_interactions,
                       item_features=item_features, num_threads=1).mean()
    rec_te = recall_at_k(model, test_interactions, train_interactions=train_interactions,
                         item_features=item_features, k=k, num_threads=1).mean()  # ⬅️ (سطر جديد) Recall@k على الاختبار

    print(f"Precision@{k}: train {prec_tr:.4f} ({prec_tr*100:.2f}%), "
          f"test {prec_te:.4f} ({prec_te*100:.2f}%)")
    print(f"Recall@{k}:    train {rec_tr:.4f} ({rec_tr*100:.2f}%), "
          f"test {rec_te:.4f} ({rec_te*100:.2f}%)  (Recommendation Accuracy)")  # ⬅️ (سطر جديد للطباعة)
    print(f"AUC:           train {auc_tr:.4f}, test {auc_te:.4f}")
