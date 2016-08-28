#include <vector>
#include <dlib/dnn/tensor.h>
#include <dlib/dnn.h>

using namespace dlib;

template <
	template<typename> class tag
	>
class mul_prev_
{
public:
	const static unsigned long id = tag_id<tag>::id;

	template <typename SUBNET>
	void setup (const SUBNET& /*sub*/)
	{
	}

	template <typename SUBNET>
	void forward(const SUBNET& sub, resizable_tensor& output)
	{
		auto&& t1 = sub.get_output();
		auto&& t2 = layer<tag>(sub).get_output();
		output.set_size(std::max(t1.num_samples(),t2.num_samples()),
						std::max(t1.k(),t2.k()),
						std::max(t1.nr(),t2.nr()),
						std::max(t1.nc(),t2.nc()));
		tt::multiply(false, output, t1, t2);
	}

	template <typename SUBNET>
	void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
	{
		// The gradient of one factor is the other factor
		tt::multiply(true, sub.get_gradient_input(), layer<tag>(sub).get_output(), gradient_input);
		tt::multiply(true, layer<tag>(sub).get_gradient_input(), sub.get_output(), gradient_input);
	}

	const tensor& get_layer_params() const { return params; }
	tensor& get_layer_params() { return params; }

	friend void serialize(const mul_prev_& , std::ostream& out)
	{
		serialize("mul_prev_", out);
	}

	friend void deserialize(mul_prev_& , std::istream& in)
	{
		std::string version;
		deserialize(version, in);
		if (version != "mul_prev_")
			throw serialization_error("Unexpected version '"+version+"' found while deserializing mul_prev_.");
	}

	friend std::ostream& operator<<(std::ostream& out, const mul_prev_& item)
	{
		out << "mul_prev"<<id;
		return out;
	}

	friend void to_xml(const mul_prev_& item, std::ostream& out)
	{
		out << "<mul_prev tag='"<<id<<"'/>\n";
	}

private:
	resizable_tensor params;
};

template <
	template<typename> class tag,
	typename SUBNET
	>
using mul_prev = add_layer<mul_prev_<tag>, SUBNET>;

/* affine layer could do the job, but I don't know how to
 * initialize it, and would waste memory holding M = -1 and
 * B = 1.
 */
class one_minus_
{
public:
	template <typename SUBNET>
    void setup (const SUBNET& /* sub */)
	{}

	void forward_inplace(const tensor& data_input, tensor& data_output)
	{
		size_t size = data_input.size();

		const float *in_data = data_input.host();
		float *out_data = data_output.host_write_only();

		for(size_t i = 0; i < size; ++i) {
			out_data[i] = 1.0f - in_data[i];
		}
	}

	void backward_inplace(
            const tensor& gradient_input,
            tensor& data_grad,
            tensor& /* params_grad */)
	{
		const float *in_data = gradient_input.host();
		float *out_data = data_grad.host();

		size_t size = gradient_input.size();

		if (is_same_object(gradient_input, data_grad)) {
			for(size_t i = 0; i < size; ++i) {
				out_data[i] = -in_data[i];
			}
		} else {
			for(size_t i = 0; i < size; ++i) {
				out_data[i] -= in_data[i];
			}
		}
	}

	const tensor& get_layer_params() const
	{
		return params;
	}

	tensor& get_layer_params(
	)
	{
		return params;
	}

	friend void serialize(const one_minus_& , std::ostream& out)
	{
		serialize("one_minus_", out);
	}

	friend void deserialize(one_minus_& , std::istream& in)
	{
		std::string version;
		deserialize(version, in);
		if (version != "one_minus_")
			throw serialization_error("Unexpected version '"+version+"' found while deserializing one_minus_.");
	}

	friend std::ostream& operator<<(std::ostream& out, const one_minus_& item)
	{
		out << "one_minus";
		return out;
	}

	friend void to_xml(const one_minus_& item, std::ostream& out)
	{
		out << "<one_minus />\n";
	}


	private:
		dlib::resizable_tensor params; // unused
};

template <typename SUBNET>
using one_minus = add_layer<one_minus_, SUBNET>;

enum split_side
{
	SPLIT_LEFT = 0,
	SPLIT_RIGHT = 1
};

template <split_side SIDE>
class split_
{
public:
	template <typename SUBNET>
    void setup (const SUBNET& sub)
	{
		auto &in = sub.get_output();
		assert(in.k() % 2 == 0);
		out_sample_size = in.k() / 2;
	}

    template <typename SUBNET>
    void forward(const SUBNET& sub, dlib::resizable_tensor& data_output)
	{
		auto &in = sub.get_output();

		size_t num_samples = in.num_samples();
		data_output.set_size(num_samples, out_sample_size, in.nr(), in.nc());

		const float *in_data = in.host();
		float *out_data = data_output.host_write_only();

		for(size_t s = 0; s < num_samples; ++s) {
			const float *in = &in_data[(s * 2 + size_t(SIDE)) * out_sample_size];
			float *out = &out_data[s * out_sample_size];
			std::copy(in, in + out_sample_size, out);
		}
	}

	template <typename SUBNET>
    void backward(
        const tensor& gradient_input,
        SUBNET& sub,
        tensor& /* params_grad */)
	{
		auto &grad_out = sub.get_gradient_input();

		const float *in_data = gradient_input.host();
		float *out_data = grad_out.host();

		size_t num_samples = gradient_input.num_samples();

		for(size_t s = 0; s < num_samples; ++s) {
			const float *in = &in_data[s * out_sample_size];
			float *out = &out_data[(s * 2 + size_t(SIDE)) * out_sample_size];
			for(size_t i = 0; i < out_sample_size; ++i) {
				out[i] += in[i];
			}
		}
	}

	const tensor& get_layer_params() const
	{
		return params;
	}

	tensor& get_layer_params()
	{
		return params;
	}

	friend void serialize(const split_& , std::ostream& out)
	{
		serialize("split_", out);
		serialize(int(SIDE), out);
	}

	friend void deserialize(split_& , std::istream& in)
	{
		std::string version;
		deserialize(version, in);
		if (version != "split_")
			throw serialization_error("Unexpected version '"+version+"' found while deserializing split_.");
		int side = 0;
		deserialize(side, in);
		if (SIDE != split_side(side))
			throw serialization_error("Wrong side found while deserializing split_");
	}

	friend std::ostream& operator<<(std::ostream& out, const split_& item)
	{
		out << "split_" << side_str();
		return out;
	}

	friend void to_xml(const split_& item, std::ostream& out)
	{
		out << "<split_" << side_str() << " />\n";
	}

private:
	static const char* side_str()
	{
		return (SIDE == SPLIT_LEFT) ? "left" : "right";
	}

	size_t out_sample_size;

	dlib::resizable_tensor params; // unused
};

template <typename SUBNET>
using split_left = add_layer<split_<SPLIT_LEFT>, SUBNET>;

template <typename SUBNET>
using split_right = add_layer<split_<SPLIT_RIGHT>, SUBNET>;

template <typename INTERNALS>
class rnn_
{
public:
	void reset_sequence()
	{
		remember_input = 0.0f;
	}

	template <typename SUBNET>
	void setup (const SUBNET& sub)
	{
		auto &in = sub.get_output();

		// Setup sequence params
		remember_input.set_size(1, in.k(), in.nr(), in.nc());
		reset_sequence();

		sample_size = in.k() * in.nr() * in.nc();
		sample_aliaser = alias_tensor(1, in.k(), in.nr(), in.nc());

		forward_nets.clear();

		trained_params.clear();
		may_have_new_params = false;

		std::cout << "Setup called!" << std::endl;
	}

	template <typename SUBNET>
	void forward(const SUBNET& sub, resizable_tensor& data_output)
	{
		//if(in_training) {
			reset_sequence();
		//}

		auto &in = sub.get_output();
		size_t num_samples = in.num_samples();

		data_output.set_size(num_samples, in.k(), in.nr(), in.nc());

		forward_nets.resize(1);
		forward_nets.reserve(num_samples);
		forward_input.clear();
		forward_input.reserve(num_samples);

		if(may_have_new_params && trained_params.size()) {
			// Current RNN layer already had its parameters updated by learning.
			// Attribute them to all children.
			visit_layer_parameters(forward_nets[0], visitor_updater(trained_params));

			may_have_new_params = false;
		}

		size_t s = 0;
		for(;;) {
			forward_input.emplace_back(1, 2*in.k(), in.nr(), in.nc());
			auto &sample_input = forward_input.back();

			// Copy the input to the first half of the tensor
			// to be together with remembered data.
			tt::copy_tensor(sample_input, 0, single_sample(in, s), 0, in.k());

			// Copy remembered data to the second half of the tensor.
			tt::copy_tensor(sample_input, sample_size, remember_input, 0, in.k());

			// Pass the new + remembered input tensor through the internal subnetwork
			auto &sout = forward_nets[s].forward(sample_input);
			assert(in.k() == sout.k() && in.nr() == sout.nr() && in.nc() == sout.nc());

			// Copy the single output to the assembled outputs
			{
				auto dest = single_sample(data_output, s);
				tt::copy_tensor(dest, 0, sout, 0, sout.k());
			}

			// Copy the single output to internal memory, to be used in next evaluation.
			tt::copy_tensor(remember_input, 0, sout, 0, in.k());

			// Test end of loop
			if(++s >= num_samples) {
				break;
			}

			// Creates the net to be used in the next iteration.
			forward_nets.emplace_back(forward_nets[s-1]);
		}
	}

	template <typename SUBNET>
    void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& params_grad_out)
	{
		auto &data_grad_out = sub.get_gradient_input();

		resizable_tensor remembered_grad(1, gradient_input.k(), gradient_input.nr(), gradient_input.nc());
		assert(sample_size == gradient_input.k() * gradient_input.nr() * gradient_input.nc());

		{
			// Sets the rembembered_grad to zero for the first iteration.
			remembered_grad = 0.0f;

			// Zeroes the params grad output before accumulating.
			params_grad_out = 0.0f;
		}

		size_t num_samples = gradient_input.num_samples();
		for(ssize_t s = num_samples - 1; s >= 0; --s) {
			// Accumulate the gradient from previous iterations with input gradient
			tt::add(remembered_grad, remembered_grad, single_sample(gradient_input, s));

			auto &fnet = forward_nets[s];
			fnet.back_propagate_error(forward_input[s], remembered_grad);

			const tensor& inner_data_grad = fnet.get_final_data_gradient();

			// Assign half of iteration's data output in the full data output.
			{
				auto dest = single_sample(data_grad_out, s);
				tt::copy_tensor(dest, 0, inner_data_grad, 0, data_grad_out.k());
			}

			// Accumulate the other half with te remembered to be used in the next iteration.
			tt::add(remembered_grad, remembered_grad, sample_aliaser(inner_data_grad, sample_size));

			// Accumulate parameters gradient
			visit_layer_parameter_gradients(forward_nets[s], visitor_accumulator(params_grad_out));
		}
	}

	const tensor& get_layer_params() const
	{
		assert(may_have_new_params);
		return trained_params;
	}

	tensor& get_layer_params()
	{
		if(!trained_params.size() && !forward_nets.empty()) {
			trained_params.clear();

			size_t counter;
			visit_layer_parameters(forward_nets[0], visitor_counter(counter));

			trained_params.set_size(1, counter);
			visit_layer_parameters(forward_nets[0], visitor_collector(trained_params));
		}

		may_have_new_params = true;
		return trained_params;
	}

	friend void serialize(const rnn_& net, std::ostream& out)
	{
		serialize("rnn_", out);
		if(net.forward_nets.empty()) {
			INTERNALS tmp;
			serialize(tmp, out);
		} else {
			serialize(net.forward_nets[0], out);
		}
	}

	friend void deserialize(rnn_& net, std::istream& in)
	{
		std::string version;
		deserialize(version, in);
		if (version != "rnn_")
			throw serialization_error("Unexpected version '"+version+"' found while deserializing rnn_.");
		if(net.forward_nets.empty()) {
			net.forward_nets.resize(1);
		}
		deserialize(net.forward_nets[0], in);
	}

	friend std::ostream& operator<<(std::ostream& out, const rnn_& net)
	{
		out << "rnn_";
		if(net.forward_nets.empty()) {
			INTERNALS tmp;
			out << tmp;
		} else {
			out << net.forward_nets[0];
		}
		return out;
	}

	friend void to_xml(const rnn_& net, std::ostream& out)
	{
		out << "<rnn>\n";
		if(net.forward_nets.empty()) {
			INTERNALS tmp;
			to_xml(tmp);
		} else {
			to_xml(net.forward_nets[0]);
		}
		to_xml(net.forward_nets[0]);
		out << "<rnn/>\n";
	}

private:
	struct visitor_counter
	{
		visitor_counter(size_t &counter):
			counter(counter)
		{
		    this->counter = 0;
		}

		void operator()(size_t, tensor& p)
		{
			counter += p.size();
		}

		size_t &counter;
	};

	class visitor_tensor
	{
	public:
		visitor_tensor(tensor& t):
			t(t),
			counter(0)
		{}

	protected:
		tensor& t;
		size_t counter;
	};

	class visitor_collector:
		public visitor_tensor
	{
	public:
		using visitor_tensor::visitor_tensor;

		void operator()(size_t, tensor& src)
		{
			auto dest = alias_tensor(src.num_samples(), src.k(), src.nr(), src.nc())(this->t, this->counter);
			memcpy(dest, src);

			this->counter += src.size();
		}
	};

	class visitor_updater:
		public visitor_tensor
	{
	public:
		using visitor_tensor::visitor_tensor;

		void operator()(size_t, tensor& dest)
		{
			auto src = alias_tensor(dest.num_samples(), dest.k(), dest.nr(), dest.nc())(this->t, this->counter);
			memcpy(dest, src);

			this->counter += dest.size();
		}
	};

	class visitor_accumulator:
		public visitor_tensor
	{
	public:
		using visitor_tensor::visitor_tensor;

		void operator()(size_t, tensor& src)
		{
			auto dest = alias_tensor(src.num_samples(), src.k(), src.nr(), src.nc())(this->t, this->counter);
			tt::add(dest, dest, src);

			this->counter += dest.size();
		}
	};

	alias_tensor_instance single_sample(tensor& t, size_t sample_idx)
	{
		return sample_aliaser(t, sample_idx * sample_size);
	}

	alias_tensor_const_instance single_sample(const tensor& t, size_t sample_idx)
	{
		return sample_aliaser(t, sample_idx * sample_size);
	}

	alias_tensor sample_aliaser;

	std::vector<INTERNALS> forward_nets;
	std::vector<dlib::resizable_tensor> forward_input;

	dlib::resizable_tensor trained_params;

	/**
	 * The data remembered from previous runs.
	 */
	dlib::resizable_tensor remember_input;

	size_t sample_size;
	size_t params_size;

	bool may_have_new_params;
};

/* An implementation of EXAMPLE_INPUT_LAYER that
 * does nothing. Meant to be used as input of
 * the inner network of rnn_.
 */
class dummy_input
{
 public:
	// sample_expansion_factor must be > 0
	const static unsigned int sample_expansion_factor = 1;
	typedef int input_type;

	template <typename forward_iterator>
	void to_tensor (forward_iterator ibegin, forward_iterator iend, resizable_tensor& data) const
	{}

    friend std::ostream& operator<<(std::ostream& out, const dummy_input& item)
    {
        out << "dummy_input";
        return out;
	}

	friend void serialize(const dummy_input& item, std::ostream& out)
	{
		serialize("dummy_input", out);
	}

    friend void deserialize(dummy_input& item, std::istream& in)
    {
        std::string version;
        deserialize(version, in);
        if (version != "dummy_input")
                throw serialization_error("Unexpected version found while deserializing dummy_input.");
	}

    friend void to_xml(const dummy_input& item, std::ostream& out)
    {
        out << "<dummy_input/>";
	}
};

// Attempt to implement the RNN MUT1, empirically found to be
// better than ordinary LSTM in "An Empirical Exploration of
// Recurrent Network Architectures" by Rafal Jozefowicz,
// Wojciech Zaremba and Ilya Sutskever, given in:
// http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf#cite.gru
//
// tag1: (x, h) (Input)
// tag2: x
// tag3: z
// tag4: h
// tag5: Whr h + Br, used for r
// tag6: Whh (r * h) + Bh
// tag7: First term of output: tanh(Whh (r * h) + tanh(x) + Bh) * z
template <unsigned long num_outputs, typename INPUT>
using inner_lstm_mut_ =
	add_prev<tag7, mul_prev<tag4, one_minus<add_skip_layer<tag3,
		tag7<mul_prev<tag3, htan<add_prev<tag6, htan<add_skip_layer<tag2, // tanh(Whh (r * h) + tanh(x) + Bh) * z, first term of output
			tag6<fc<num_outputs, mul_prev<tag4, sig<add_prev<tag5, fc_no_bias<num_outputs, add_skip_layer<tag2, // Whh (sig(Whr h + Wxr x + Br) * h) + Bh
				tag5<fc<num_outputs, // Whr h + Br, used in r
					tag4<split_right<add_skip_layer<tag1, // h
						tag3<sig<fc<num_outputs, // z
							tag2<split_left< // x
								tag1<INPUT> // Input
							>>
						>>>
					>>>
				>>
			>>>>>>>
		>>>>>>
	>>>>;

template <unsigned long num_outputs, typename SUBNET>
using lstm_mut = add_layer<rnn_<inner_lstm_mut_<num_outputs, dummy_input>>, SUBNET>;
