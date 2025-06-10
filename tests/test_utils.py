import NepTrainKit.utils as utils

def test_parse_index_string():
    assert utils.parse_index_string('1:4', 10) == [1,2,3]
    assert utils.parse_index_string(':3', 5) == [0,1,2]
    assert utils.parse_index_string('::2', 5) == [0,2,4]
