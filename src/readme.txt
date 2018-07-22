install gsl lib
g++ -Wall -std=c++11 main.cpp -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF -lgsl -lgslcblas -lm lpthread -o ctm

input dataformat:
each line is a document with:
doc_id word_id:count word_id:count ...
