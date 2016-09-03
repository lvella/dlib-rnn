#FLAGS = -O3 -flto -march=native -mtune=native
FLAGS = -g

rnn-sample: rnn-sample.o
	g++ -o rnn-sample rnn-sample.o `pkg-config dlib-1 --libs` -std=c++14 $(FLAGS)

rnn-sample.o: input_one_hot.h rnn.h rnn-sample.cpp
	g++ -c rnn-sample.cpp `pkg-config dlib-1 --cflags` -std=c++14 $(FLAGS)

clean:
	rm *.o rnn-sample
