import unittest
import pandas as pd
import numpy as np

from src.preprocessing import (
    rule1_remove_exact_duplicates,
    rule2_filter_bad_debt_and_adjustments,
    rule3_filter_non_product_codes,
    rule4_tag_cancellations_and_revenue,
    rule5_standardize_descriptions,
    rule6_split_store_and_customer_data,
    EXCLUDED_STOCK_CODES,
)


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "InvoiceNo": ["536365", "536365", "C536379", "536380", "536381"],
            "StockCode": ["85123A", "85123A", "D", "85123A", "POST"],
            "Description": ["White heart light", "White heart light", "Discount", " white heart light ", "Postage"],
            "Quantity": [6, 6, -1, 10, 1],
            "InvoiceDate": ["2010-12-01 08:26:00"] * 5,
            "UnitPrice": [2.55, 2.55, 27.50, 2.55, 18.00],
            "CustomerID": [17850.0, 17850.0, 17850.0, np.nan, 13047.0],
            "Country": ["United Kingdom"] * 5,
        })

    def test_rule1_remove_exact_duplicates(self):
        dedup_df, dropped = rule1_remove_exact_duplicates(self.sample_df)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(dedup_df), 4)

    def test_rule2_filter_bad_debt_and_adjustments(self):
        test_df = pd.DataFrame({
            "InvoiceNo": ["536365", "A563185", "C536367", "536368", "536369"],
            "Quantity": [5, 1, -2, 0, -3],
            "UnitPrice": [2.50, -11062.06, 3.00, 2.00, 0.0],
        })
        clean_df, bad_debt, invalid_qp = rule2_filter_bad_debt_and_adjustments(test_df)
        self.assertEqual(bad_debt, 1)
        self.assertEqual(invalid_qp, 2)
        self.assertEqual(len(clean_df), 2)
        self.assertTrue((clean_df["UnitPrice"] > 0).all())

    def test_rule3_filter_non_product_codes(self):
        test_df = pd.DataFrame({
            "StockCode": ["85123A", "POST", "22423", "BANK CHARGES", "M"],
        })
        filtered_df, dropped = rule3_filter_non_product_codes(test_df)
        self.assertEqual(len(filtered_df), 2)
        self.assertEqual(dropped, 3)
        self.assertFalse(filtered_df["StockCode"].isin(EXCLUDED_STOCK_CODES).any())

    def test_rule4_tag_cancellations_and_revenue(self):
        test_df = pd.DataFrame({
            "InvoiceNo": ["536365", "C536366"],
            "Quantity": [10, -2],
            "UnitPrice": [2.50, 5.00],
        })
        tagged_df = rule4_tag_cancellations_and_revenue(test_df)
        self.assertIn("is_cancellation", tagged_df.columns)
        self.assertIn("TransactionType", tagged_df.columns)
        self.assertIn("LineTotal", tagged_df.columns)
        self.assertEqual(list(tagged_df["is_cancellation"]), [False, True])
        self.assertEqual(list(tagged_df["TransactionType"]), ["Sale", "Cancelled/Returned"])
        self.assertEqual(list(tagged_df["LineTotal"]), [25.0, -10.0])
        self.assertEqual(tagged_df["LineTotal"].sum(), 15.0)

    def test_rule5_standardize_descriptions(self):
        test_df = pd.DataFrame({
            "StockCode": ["85123A", "85123A", "85123A", "85123A"],
            "Description": ["WHITE HANGING HEART", " white hanging heart ", "WHITE HANGING HEART", np.nan],
        })
        standardized_df = rule5_standardize_descriptions(test_df)
        self.assertTrue((standardized_df["Description"] == "WHITE HANGING HEART").all())

    def test_rule6_split_store_and_customer_data(self):
        test_df = pd.DataFrame({
            "InvoiceNo": ["536365", "536366"],
            "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:28:00"],
            "CustomerID": [17850.0, np.nan],
        })
        store_df, customer_df = rule6_split_store_and_customer_data(test_df)
        self.assertEqual(len(store_df), 2)
        self.assertEqual(len(customer_df), 1)
        self.assertEqual(store_df.loc[0, "CustomerID"], "17850")
        self.assertTrue(store_df.loc[1, "is_guest"])
        self.assertFalse(customer_df.loc[0, "is_guest"])


if __name__ == "__main__":
    unittest.main()
