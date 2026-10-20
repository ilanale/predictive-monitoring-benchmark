# Modified by ilanale/predictive-monitoring-benchmark for court-case adaptation — see CHANGES_FROM_UPSTREAM.md
import EncoderFactory

from bucketers.ClusterBasedBucketer import ClusterBasedBucketer
from bucketers.StateBasedBucketer import StateBasedBucketer
from bucketers.PrefixLengthBucketer import PrefixLengthBucketer
from bucketers.NoBucketer import NoBucketer
# compat: bucketers/KNNBucketer.py is not in the repository; imported lazily below (KNN is out of scope)
from sklearn.cluster import KMeans

        
def get_bucketer(method, encoding_method=None, case_id_col=None, cat_cols=None, num_cols=None, n_clusters=None, random_state=None, n_neighbors=None):

    if method == "cluster":
        bucket_encoder = EncoderFactory.get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        clustering = KMeans(n_clusters, random_state=random_state, n_init=10)  # compat: sklearn>=1.4 default n_init='auto'; keep old 10
        return ClusterBasedBucketer(encoder=bucket_encoder, clustering=clustering)
        
    elif method == "state":
        bucket_encoder = EncoderFactory.get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        return StateBasedBucketer(encoder=bucket_encoder)
            
    elif method == "single":
        return NoBucketer(case_id_col=case_id_col)

    elif method == "prefix":
        return PrefixLengthBucketer(case_id_col=case_id_col)
    
    elif method == "knn":
        from bucketers.KNNBucketer import KNNBucketer  # compat: lazy import, module missing upstream
        bucket_encoder = EncoderFactory.get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        return KNNBucketer(encoder=bucket_encoder, n_neighbors=n_neighbors)

    else:
        print("Invalid bucketer type")
        return None