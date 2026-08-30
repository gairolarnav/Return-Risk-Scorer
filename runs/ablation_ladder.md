| feature_set                              |   n_features |   macro_f1 |   mean_top_prob |   threshold_sensitive_frac |
|:-----------------------------------------|-------------:|-----------:|----------------:|---------------------------:|
| A. all features                          |           35 |     0.9990 |          0.9998 |                     0.0007 |
| B. -wishlist_to_cart_time_hrs            |           34 |     0.9936 |          0.9992 |                     0.0023 |
| C. B -days_to_return                     |           33 |     0.9779 |          0.9950 |                     0.0160 |
| D. C -return_rate_pct                    |           31 |     0.9767 |          0.9937 |                     0.0196 |
| E. D -total_returns_lifetime             |           29 |     0.9586 |          0.9879 |                     0.0373 |
| F. E -customer_support_contacts          |           28 |     0.9342 |          0.9717 |                     0.0843 |
| G. F -previous_dispute_count  <- TESTBED |           27 |     0.8993 |          0.9460 |                     0.1607 |
| H. G -avg_order_value, -refund_amount    |           24 |     0.8581 |          0.9186 |                     0.2394 |
