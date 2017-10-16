The following structure is adhered to in the indexing:

{
    "N": int,
    "papers": {
        $field: {
            $paper_id: {
                "number_of_terms": int,
                "number_of_unique_terms": int,
                "tf": dict,
                "wf": dict,
                "tf.idf": dict,
                "wf.idf": dict,
                "vector_lengths": {
                    "tf": int,
                    "wf": float,
                    "tf.idf": float,
                    "wf.idf": float
                }
            }
        }
    },
    "collection": {
        $field: {
            "number_of_terms": int,
            "number_of_unique_terms": int,
            "cf": dict,
            "df": dict,
            "idf": dict,
            "vector_lengths": {
                "cf": int,
                "df": int,
                "idf": float
            }
        }
    }
}