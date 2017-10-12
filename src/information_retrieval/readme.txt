The following structure is adhered to in the indexing:

{
    "N": int,
    "papers": {
        $paper_id: {
            $field: {
                "number_of_terms": int,
                "number_of_unique_terms": int,
                "frequencies": { -> This is a DataFrame
                    "tf", -> so a specific term can be found with ["tf"][$term]
                    "wf"
                },
                "vector_lengths": {
                    "tf": int
                    "wf": float
                }
            }
        }
    },
    "collection": {
        $field: {
            "number_of_terms": int,
            "number_of_unique_terms": int,
            "frequencies": { -> This is a DataFrame
                "cf", 
                "df",
                "idf"
            },
            "vector_lengths": {
                "cf": int
                "df": int
                "idf": float
            }
        }
    }
}