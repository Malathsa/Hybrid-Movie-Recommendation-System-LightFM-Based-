from lightfm import LightFM
import time
import pickle


print(f"\n🔍 Final check:")
print(f"   all_items: {len(all_items)} movies")
print(f"   item_features: {item_features.shape[0]} rows")
print(f"   Match: {len(all_items) == item_features.shape[0]}")

if len(all_items) == item_features.shape[0]:
    print("✅ READY FOR HYBRID RECOMMENDATIONS!")
else:
    print("❌ NEED TO FIX ITEM FEATURES!")


train_df, test_df = train_test_split(
    positive,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining data: {len(train_df)} interactions")
print(f"Testing data: {len(test_df)} interactions")

# Ensure test data exists in training
train_users = set(train_df["userId"].unique())
train_items = set(train_df["movieId"].unique())
test_df = test_df[
    test_df["userId"].isin(train_users) & 
    test_df["movieId"].isin(train_items)
].copy()

print(f"Testing data after filtering: {len(test_df)} interactions")

def prepare_interactions(df):
    return dataset.build_interactions(
        [(row.userId, row.movieId) for row in df.itertuples(index=False)]
    )[0]

train = prepare_interactions(train_df)
test = prepare_interactions(test_df)

print(f"Train interactions shape: {train.shape}")
print(f"Test interactions shape: {test.shape}")
print(f"Item features shape: {item_features.shape}")

print("\n" + "="*50)
print("Hybrid Model Training (WARP + IDF-Weighted Genres)")
print("="*50)

model_hybrid = LightFM(
    loss="warp",
    learning_rate=0.05,
    random_state=42
)

# Train with IDF-weighted content features
model_hybrid.fit(
    train,
    item_features=item_features,
    epochs=15,
    num_threads=1
)

# Evaluate hybrid model
evaluate_model(model_hybrid, train, test, item_features, k=10)

# ===== 11a) Save Model Checkpoint =====
with open("lightfm_hybrid_checkpoint.pkl", "wb") as f:
    pickle.dump(model_hybrid, f)

print("💾 Saved model checkpoint to 'lightfm_hybrid_checkpoint.pkl'")
