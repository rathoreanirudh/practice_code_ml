import app


def test_answer():
    assert app.generate_annotation_id() >= 6000
    assert app.generate_annotation_id() <= 9000
