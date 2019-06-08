OPT_FLAGS = -Ofast -flto=8 -fno-fat-lto-objects -march=native -mtune=native
#OPT_FLAGS = -g

CFLAGS = -std=c++17 `pkg-config dlib-1 --cflags` -pthread $(OPT_FLAGS)
LDFLAGS = $(CFLAGS) `pkg-config dlib-1 --libs` -lopenblas

rnn-sample: rnn-sample.o
	g++ -o rnn-sample rnn-sample.o $(LDFLAGS)

rnn.h.gch: rnn.h
	g++ -c rnn.h $(CFLAGS)

rnn-sample.o: input_one_hot.h rnn.h.gch rnn-sample.cpp
	g++ -c rnn-sample.cpp $(CFLAGS)

clean:
	rm *.o rnn-sample rnn.h.gch
