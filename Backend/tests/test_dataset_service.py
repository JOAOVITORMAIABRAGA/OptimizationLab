from io import BytesIO

import pandas as pd
import pytest

from services.dataset_service import DatasetService


def test_summarize_many_supports_multiple_formats():
    service = DatasetService()

    workbook = BytesIO()
    pd.DataFrame({"product": ["A", "B"], "capacity": [10, 20]}).to_excel(
        workbook,
        index=False,
        engine="openpyxl",
    )

    result = service.summarize_many(
        [
            ("products.csv", b"product,cost\nA,10\nB,20\n"),
            ("limits.txt", b"product\tcapacity\nA\t10\nB\t20\n"),
            ("parameters.xlsx", workbook.getvalue()),
        ]
    )

    assert result["dataset_count"] == 3
    assert [item["format"] for item in result["datasets"]] == ["csv", "txt", "xlsx"]
    assert result["datasets"][1]["columns"] == ["product", "capacity"]
    assert result["datasets"][2]["sheet_count"] == 1


def test_rejects_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        DatasetService().summarize(b"hello", "data.json")


def test_rejects_more_than_ten_files():
    files = [(f"data_{index}.csv", b"id\n1\n") for index in range(11)]
    with pytest.raises(ValueError, match="maximum of 10"):
        DatasetService().summarize_many(files)
