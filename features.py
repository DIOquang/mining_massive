from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# Trỏ thẳng vào thư mục Parquet trên GCS của bạn
user_stats_source = FileSource(
    path="gs://amazon-reviews-lakehouse-warehouse/warehouse/gold/user_features/",
    timestamp_field="event_timestamp",
)

# SỬA LỖI Ở ĐÂY: Dùng ValueType.STRING chuẩn của Feast thay vì String
user = Entity(name="user_id", join_keys=["user_id"], value_type=ValueType.STRING)

user_stats_view = FeatureView(
    name="user_statistical_features",
    entities=[user],
    ttl=timedelta(days=3650), # Lịch sử dài hạn
    schema=[
        Field(name="total_reviews", dtype=Int64),
        Field(name="avg_rating_given", dtype=Float32),
        Field(name="stddev_rating_given", dtype=Float32),
    ],
    source=user_stats_source,
)