#include <fenv.h>

#include <vector>
#include <fstream>

#include "rnn.h"
#include "input_one_hot.h"

int main()
{
	feenableexcept(FE_INVALID   |
		FE_DIVBYZERO);

	std::vector<char> input;
	std::vector<unsigned long> labels;
	{
		std::ifstream fd("tiny-shakespeare.txt");
		fd.seekg(0, std::ios_base::end);
		unsigned fileSize = fd.tellg();
	    input.resize(fileSize - 1u);
		labels.resize(fileSize - 2u);

		fd.seekg(0, std::ios_base::beg);
		fd.read(&input[0], fileSize - 1);
		std::copy(input.begin() + 1, input.end(), labels.begin());

		char last;
		fd.read(&last, 1);
		labels.push_back(last);
	}

	unsigned label_count;
	int label_map[256];
	char fmap[65];
	std::fill(&label_map[0], &label_map[256], -1);

	label_map[input[0]] = 0;
	input[0] = 0;
	label_count = 1;
	for(unsigned i = 1; i < input.size(); ++i) {
		unsigned label;
		if(label_map[input[i]] < 0) {
			label = label_count++;
		} else {
			label = label_map[input[i]];
		}
		label_map[input[i]] = label;
		fmap[label] = input[i];
		input[i] = label;
		labels[i-1] = label;
	}
	if(label_map[labels.back()] < 0) {
		label_map[labels.back()] = label_count;
		labels.back() = label_count;
		++label_count;
	} else {
		labels.back() = label_map[labels.back()];
	}
	assert(label_count == 65);

	using net_type =
		loss_multiclass_log<
		// Someone told me on IRC that using a softmax layer improves training...
		softmax<
		fc<65,
		lstm_mut<100,
		fc<100,
		input_one_hot<char, 65>
	>
	>>>>;

	net_type net;

	dnn_trainer<net_type, adam> trainer(net, adam(0.0005, 0.9, 0.999));

	trainer.set_mini_batch_size(10);

	// The training rate must be much smaller because the number of
	// gradients accumulated for each parameter is:
	// (s + s²) / 2 * o
	// where s is the sequence size (mini batch size), and o is the output
	// size, where on a feedforward network, the number of gradients
	// accumulated for each parameter is simply:
	// s * o
	trainer.set_learning_rate(1e-4);
	trainer.set_learning_rate_shrink_factor(0.85);
	trainer.set_min_learning_rate(1e-10);
	trainer.set_iterations_without_progress_threshold(10000);

	trainer.be_verbose();

	//trainer.set_synchronization_file("shakespeare.sync", std::chrono::seconds(20));
	trainer.train(input, labels);

    net.clean();
    serialize("shakespeare_network.dat") << net;

    // Drop loss layer
    /*auto &generator = net.subnet();
    char prev = '\n';
    for(unsigned i = 0; i < 500; ++i) {
	    auto &l = generator(prev);
	    const float *h = l.host();
	    unsigned m = 0;
	    double sum = h[0];
	    for(unsigned j = 1; j < 65; ++j) {
		if(h[j] > h[m])
		    m = j;
	    }
	    std::cout << (prev = fmap[m]);
    }
    std::cout << std::endl;*/
}
