from camp.models.recommender import AttentionRecommender
from camp.train.recommender import AttentionRecommenderTrainer


def test_recommender_keeps_forward_and_drops_train_infer():
    assert "fit_trainer" not in AttentionRecommender.__dict__
    assert "recommend_placements" not in AttentionRecommender.__dict__
    assert not hasattr(AttentionRecommender, "fit_trainer")
    assert not hasattr(AttentionRecommender, "recommend_placements")
    assert AttentionRecommenderTrainer.__name__ == "AttentionRecommenderTrainer"
