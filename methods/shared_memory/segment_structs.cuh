#ifndef SEGMENT_STRUCT
#define SEGMENT_STRUCT
typedef struct segment {
	WORD_TYPE l_end_type;
	int l_end_len;

	WORD_TYPE r_end_type;
	int r_end_len;
} segment;
#endif // SEGMENT_STRUCT

#ifndef SEGMENT_SOA_STRUCT
#define SEGMENT_SOA_STRUCT
typedef struct segment_soa {
	WORD_TYPE* l_end_type;
	int* l_end_len;

	WORD_TYPE* r_end_type;
	int* r_end_len;
} segment_soa;
#endif // SEGMENT_SOA_STRUCT
