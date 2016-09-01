rnn-sample: input_one_hot.h rnn.h rnn-sample.cpp
	g++ -o rnn-sample rnn-sample.cpp `pkg-config dlib-1 --libs --cflags` -std=c++14 -O3 -flto -march=native -mtune=native

clean:
	rm rnn-sample
