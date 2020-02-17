from app import generate_annotation_id


def test_answer():
    assert generate_annotation_id() >= 6000
    assert generate_annotation_id() <= 9000
