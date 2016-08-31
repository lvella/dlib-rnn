#include <fenv.h>

#include <vector>
#include <fstream>
#include <random>

#include "rnn.h"
#include "input_one_hot.h"

template <typename SUBNET>
using rnn_type_h = lstm_mut1<512, SUBNET>;

template <typename SUBNET>
using rnn_type_l = lstm_mut3<512, SUBNET>;


using net_type =
	loss_multiclass_log<
	fc<65,
	rnn_type_h<
	rnn_type_l<
	input_one_hot<char, 65>
>>>>;


void train(std::vector<char>& input, std::vector<unsigned long>& labels)
{
	net_type net;
	dnn_trainer<net_type, adam> trainer(net, adam(0.0005, 0.9, 0.999));

	//trainer.set_mini_batch_size(70);

	trainer.set_learning_rate_shrink_factor(0.6);
	//trainer.set_learning_rate(0.9);
	//trainer.set_min_learning_rate(1e-9);
	//trainer.set_learning_rate_shrink_factor(0.4);
	//trainer.set_iterations_without_progress_threshold(10000);

	trainer.be_verbose();

	trainer.set_synchronization_file("shakespeare.sync", std::chrono::seconds(120));

	std::mt19937 gen(std::random_device{}());
	std::uniform_int_distribution<unsigned> size(10, 350);
	std::uniform_int_distribution<unsigned> start(0, input.size() - 11);

	std::vector<char> b_input;
	std::vector<unsigned long> b_labels;
	do {
		unsigned i = start(gen);
		unsigned e = i + size(gen);
		if(e > input.size() - 1) {
			e = input.size() - 1;
		}

		b_input.resize(e - i);
		std::copy(input.begin() + i, input.begin() + e, b_input.begin());

		b_labels.resize(e - i);
		std::copy(labels.begin() + i, labels.begin() + e, b_labels.begin());

        	trainer.train_one_step(b_input, b_labels);
	} while(trainer.get_learning_rate() >= 1e-7);

	net.clean();
	serialize("shakespeare_network.dat") << net;

}

void run_test(char fmap[])
{
	// Net with loss layer replaced with softmax
	softmax<net_type::subnet_type> generator;

	// Load the network
	{
		net_type net;

		try {
			// First, try to open the fully trained network
			deserialize("shakespeare_network.dat") >> net;
			std::cerr << "Using fully trained network." << std::endl;
		} catch(dlib::serialization_error&) {
			// Failing, try to open the partially trained network
			dnn_trainer<net_type, adam> trainer(net);
			trainer.set_synchronization_file("shakespeare.sync");
			std::cerr << "Using partially trained network." << std::endl;
		}

		// Replace loss layer with softmax
		generator.subnet() = net.subnet();
	}

	// Configure to not forget between evaluations
	layer<rnn_type_h>(generator).layer_details()
		.set_batch_is_full_sequence(false);
	layer<rnn_type_l>(generator).layer_details()
		.set_batch_is_full_sequence(false);

	std::random_device rd;

	// Start generation from newline character
	char prev = '\n';
	for(unsigned i = 0; i < 100; ++i) {
		double rnd =
			std::generate_canonical<double, std::numeric_limits<double>::digits>(rd);

		auto &l = generator(prev);
		const float *h = l.host();
		double sum = 0.0f;
		unsigned j;
		for(j = 0; j < 64; ++j) {
			sum += h[j];
			if(rnd <= sum)
				break;
		}
		std::cout << (prev = fmap[j]);
	}
	std::cout << std::endl;
}

int main(int argc, char *argv[])
{
	/* Better die than contaminate the sync file. */
	/*feenableexcept(FE_INVALID |
		FE_DIVBYZERO |
		FE_OVERFLOW  |
		FE_UNDERFLOW
	);*/

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

	net_type net;

	if(argc > 1) {
		if(strcmp(argv[1], "sample") == 0) {
			std::cerr << "Sampling.\n" << std::endl;
			run_test(fmap);
			return 0;
		} else if (strcmp(argv[1], "train") != 0) {
			std::cerr << "Error: invalid command.\nChoose one of\n " << argv[0] << " sample\n " << argv[0] << " train" << std::endl;
			return -1;
		}
	}

	std::cerr << "Training.\n" << std::endl;
	train(input, labels);
}

