#include <fenv.h>

#include <vector>
#include <fstream>
#include <random>

#include "rnn.h"
#include "input_one_hot.h"

template <typename V, typename N>
auto visit_rnns(V visitor, N& net, int)
	->decltype(net.subnet(), void())
{
	visit_rnns(visitor, net.subnet());
}

template <typename V, typename N>
void visit_rnns(V visitor, N& net, long)
{}

template <typename V, typename N>
void visit_rnns(V visitor, N& net)
{
	visit_rnns(visitor, net, 0);
}

template <typename V, typename SUBNET, typename INNER,
	size_t K, size_t NR, size_t NC>
void visit_rnns(V visitor, add_layer<rnn_<INNER, K, NR, NC>, SUBNET>& net)
{
	visitor(net.layer_details());
	visit_rnns(visitor, net.subnet());
}

// TODO: handle repeat...

const unsigned seq_size = 256;
const unsigned mini_batch_size = 100;
const unsigned ab_size = 64;

using net_type =
	loss_multiclass_log<
	fc<ab_size,
	lstm_mut1<256,
	lstm_mut1<256,
	fc<256,
	input_one_hot<char, ab_size>
>>>>>;

void train(std::vector<char>& input, std::vector<unsigned long>& labels)
{
	net_type net;
	dnn_trainer<net_type, adam> trainer(net, adam(0.0005, 0.9, 0.999));

	visit_rnns([](auto& n) { n.set_mini_batch_size(mini_batch_size); }, net);

	trainer.set_learning_rate_shrink_factor(0.5);
	trainer.set_learning_rate(0.01);
	//trainer.set_min_learning_rate(1e-9);
	//trainer.set_learning_rate_shrink_factor(0.4);
	trainer.set_iterations_without_progress_threshold(200);

	trainer.be_verbose();

	trainer.set_synchronization_file("shakespeare.sync", std::chrono::seconds(120));

	std::vector<unsigned> slices(input.size() / seq_size);
	assert(slices.size() >= mini_batch_size);
	for(unsigned i = 0; i < slices.size(); ++i) {
		slices[i] = i * seq_size;
	}

	std::mt19937 gen(std::random_device{}());
	std::shuffle(slices.begin(), slices.end(), gen);

	std::vector<char> b_input;
	std::vector<unsigned long> b_labels;
	for(;;) { // Reductions
		unsigned j = 0;
		while(j < slices.size()) { // Full epoch
			// Calculate the biggest sequence size we can use
			unsigned min_seq = seq_size;
			for(unsigned b = 0; b < mini_batch_size; ++b) {
				unsigned ss = slices[(j + b) % slices.size()];
				unsigned size = input.size() - ss;
				if(min_seq > size) {
					min_seq = size;
					std::cout << "DEBUG: sequence size = " << min_seq << std::endl;
					break;
				}
			}
			b_input.resize(min_seq * mini_batch_size);
			b_labels.resize(b_input.size());

			transpose_iterator<decltype(b_input.begin())> input_iter(b_input.begin(), b_input.end(), mini_batch_size);
			transpose_iterator<decltype(b_labels.begin())> label_iter(b_labels.begin(), b_labels.end(), mini_batch_size);
			for(unsigned b = 0; b < mini_batch_size; ++b) {
				unsigned ss = slices[j++ % slices.size()];
				std::copy(&input[ss], &input[ss] + min_seq, input_iter + b * min_seq);
				std::copy(&labels[ss], &labels[ss] + min_seq, label_iter + b * min_seq);
			}
			trainer.train_one_step(b_input, b_labels);
		}
		// Shuffle per epoch.
		std::shuffle(slices.begin(), slices.end(), gen);
	}
	trainer.get_net();
	net.clean();
	serialize("shakespeare_network.dat") << net;
}

void run_test(int label_map[], char fmap[])
{
	// Net with loss layer replaced with softmax
	softmax<net_type::subnet_type> generator;

	// Load the network
	{
		net_type net;

		try {
			// First, try to open the fully trained network
			deserialize("shakespeare_network.dat") >> net;
			std::cerr << "Using fully trained network.\n" << std::endl;
		} catch(dlib::serialization_error&) {
			// Failing, try to open the partially trained network
			dnn_trainer<net_type, adam> trainer(net);
			trainer.set_synchronization_file("shakespeare.sync");
			std::cerr << "Using partially trained network.\n" << std::endl;
			trainer.get_net();
		}

		// Replace loss layer with softmax
		generator.subnet() = net.subnet();
	}

	// Configure to evaluate, not to train
	visit_rnns([](auto& n) { n.set_for_run(); }, generator);

	std::random_device rd;

	// Start generation from newline character
	int prev = label_map['\n'];
	for(unsigned i = 0; i < 3000; ++i) {
		double rnd =
			std::generate_canonical<double, std::numeric_limits<double>::digits>(rd);

		auto &l = generator(prev);
		const float *h = l.host();
		double sum = 0.0f;
		unsigned j;
		for(j = 0; j < ab_size - 1; ++j) {
			sum += h[j];
			if(rnd <= sum)
				break;
		}
		std::cout << fmap[j];
		prev = j;
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
	char fmap[ab_size];
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
	assert(label_count == ab_size);

	net_type net;

	if(argc > 1) {
		if(strcmp(argv[1], "sample") == 0) {
			std::cerr << "Sampling." << std::endl;
			run_test(label_map, fmap);
			return 0;
		} else if (strcmp(argv[1], "train") != 0) {
			std::cerr << "Error: invalid command.\nChoose one of\n " << argv[0] << " sample\n " << argv[0] << " train" << std::endl;
			return -1;
		}
	}

	std::cerr << "Training." << std::endl;
	train(input, labels);
}

