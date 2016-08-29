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
		auto &prev_layer = layer<tag>(sub);
		tt::multiply(true, sub.get_gradient_input(), prev_layer.get_output(), gradient_input);
		tt::multiply(true, prev_layer.get_gradient_input(), sub.get_output(), gradient_input);
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


/* Simply forwards a previously set vector, ignoring the layer below.
   Again, affine layer could do it, if A = 0.
*/
class constant_
{
public:
	template <typename SUBNET>
	void setup (const SUBNET& /* sub */)
	{}

	void set_constant(const tensor& c)
	{
		gradient.copy_size(c);
		output = c;
	}

	const tensor &get_data_gradient()
	{
		return gradient;
	}

	template <typename SUBNET>
	void forward(const SUBNET&, resizable_tensor& data_output)
	{
		data_output.copy_size(output);
		memcpy(data_output, output);
	}

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET&/* sub */, tensor&/* params_grad */)
	{
		memcpy(gradient, gradient_input);
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

	friend void serialize(const constant_& net, std::ostream& out)
	{
		serialize("constant_", out);
		serialize(net.output, out);
		serialize(net.gradient, out);
	}

	friend void deserialize(constant_& net, std::istream& in)
	{
		std::string version;
		deserialize(version, in);
		if (version != "constant_")
			throw serialization_error("Unexpected version '"+version+"' found while deserializing constant_.");
		deserialize(net.output, in);
		deserialize(net.gradient, in);
	}

	friend std::ostream& operator<<(std::ostream& out, const constant_& item)
	{
		out << "constant";
		return out;
	}

	friend void to_xml(const constant_& item, std::ostream& out)
	{
		out << "<constant />\n";
	}


	private:
		dlib::resizable_tensor params; // unused

		dlib::resizable_tensor output;
		dlib::resizable_tensor gradient;
};

template <typename SUBNET>
using constant = add_layer<constant_, SUBNET>;


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

// This is the subnet to be used as input for
// the internal RNN implementation. The output
// of this subnet is the remembered value from
// previous sample. The current input can be
// obtained with skip_rnn_input skip layer.
// For convenience there is also a skip_rnn_memory
// giving access the remembered value.
using rnn_subnet_base =
	add_tag_layer<99992,
	constant<
	add_tag_layer<99991,
	dummy_input
	>>>;

template <typename SUBNET> using tag_rnn_input = add_tag_layer<99991, SUBNET>;
template <typename SUBNET> using tag_rnn_memory = add_tag_layer<99992, SUBNET>;

template <typename SUBNET> using skip_rnn_input = add_skip_layer<tag_rnn_input, SUBNET>;
template <typename SUBNET> using skip_rnn_memory = add_skip_layer<tag_rnn_memory, SUBNET>;

template <typename INTERNALS, size_t memory_k_, size_t memory_nr_ = 1, size_t memory_nc_ = 1>
class rnn_
{
public:
	rnn_(
		size_t mem_k  = memory_k_,
		size_t mem_nr = memory_nr_,
		size_t mem_nc = memory_nc_
	):
		batch_is_full_sequence(true),
		out_sample_size(mem_k * mem_nr * mem_nc),
		out_sample_aliaser(1, mem_k, mem_nr, mem_nc)
	{
		remember_input.set_size(1, mem_k, mem_nr, mem_nc);
	}

	rnn_(const rnn_&) = default;

	void reset_sequence()
	{
		remember_input = 0.0f;
	}

	void set_batch_is_full_sequence(bool is_full_sequence)
	{
		batch_is_full_sequence = is_full_sequence;
	}

	template <typename SUBNET>
	void setup (const SUBNET& sub)
	{
		auto &in = sub.get_output();

		// Setup sequence params
		remember_input.set_size(1, in.k(), in.nr(), in.nc());
		reset_sequence();

		in_sample_size = in.k() * in.nr() * in.nc();
		in_sample_aliaser = alias_tensor(1, in.k(), in.nr(), in.nc());

		forward_nets.clear();

		trained_params.clear();
		may_have_new_params = false;
	}

	template <typename SUBNET>
	void forward(const SUBNET& sub, resizable_tensor& data_output)
	{
		if(batch_is_full_sequence) {
			reset_sequence();
		}

		auto &in = sub.get_output();
		size_t num_samples = in.num_samples();

		data_output.set_size(num_samples, in.k(), in.nr(), in.nc());

		forward_nets.resize(1);
		forward_nets.reserve(num_samples);

		if(may_have_new_params && trained_params.size()) {
			// Current RNN layer already had its parameters updated by learning.
			// Attribute them to all children.
			visit_layer_parameters(forward_nets[0], visitor_updater(trained_params));

			may_have_new_params = false;
		}

		const tensor *remembered = &remember_input;
		size_t s = 0;
		for(;;) {
			// Get the input tensor
			auto sample_input = in_sample_aliaser(in, s * in_sample_size);

			// Copy the remembered data to the inner network's
			// special memory layer. Every network buit upon
			// rnn_subnet_base will have it.
			auto& mem_layer = get_memory_layer(forward_nets[s]);
			mem_layer.set_constant(*remembered);

			// Pass the input tensor through the internal subnetwork
			auto &sout = forward_nets[s].forward(sample_input);

			// Copy the single output to the assembled outputs
			{
				auto dest = out_sample_aliaser(data_output, s * out_sample_size);
				memcpy(dest, sout);
			}

			// Test end of loop
			if(++s >= num_samples) {
				// Copy the single output to internal memory, to be used in next evaluation.
				memcpy(remember_input, sout);

				break;
			}

			// Use the output of this iteration as memory input of the next
			remembered = &sout;

			// Creates the net to be used in the next iteration.
			forward_nets.emplace_back(forward_nets[s-1]);
		}
	}

	template <typename SUBNET>
	void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& params_grad_out)
	{
		auto &data_grad_out = sub.get_gradient_input();
		auto &in = sub.get_output();

		resizable_tensor remembered_grad(1, gradient_input.k(), gradient_input.nr(), gradient_input.nc());
		assert(out_sample_size == gradient_input.k() * gradient_input.nr() * gradient_input.nc());

		{
			// Zeroes the params grad output before accumulating.
			params_grad_out = 0.0f;
		}

		// Get the loop counter, backwards because this is a BACKpropagation.
		size_t s = gradient_input.num_samples() - 1;

		// Get the gradient slice used in the first iteration
		auto in_grad_slice = out_sample_aliaser(gradient_input, s * out_sample_size);
		const tensor *grad_input = &in_grad_slice.get();
		for(;;) {
			// Retrieve the interation's network
			auto &fnet = forward_nets[s];

			// Get the input tensor, the same use in forward operation
			auto sample_input = in_sample_aliaser(in, s * in_sample_size);

			// Do the backpropagation in the inner network and get the output.
			fnet.back_propagate_error(sample_input, *grad_input);
			const tensor& inner_data_grad = fnet.get_final_data_gradient();

			// Assign iteration data grad output in the full data output.
			{
				auto dest = in_sample_aliaser(data_grad_out, s * in_sample_size);
				memcpy(dest, inner_data_grad);
			}

			// Accumulate parameters gradient
			visit_layer_parameter_gradients(fnet, visitor_accumulator(params_grad_out));

			// Test loop end
			if(s-- == 0) {
			    break;
			}

			// Prepare the input gradient for the next iteration.
			// It is the sum of the memory gradient output by this iteration,
			// with the corresponding slice of gradient input.

			// Get the input slice
			in_grad_slice = out_sample_aliaser(gradient_input, s * out_sample_size);

			// Get the memory gradient
			const tensor& mem_grad = get_memory_layer(fnet).get_data_gradient();

			// Add both together:
			tt::add(remembered_grad, in_grad_slice, mem_grad);

			// Set the sum as grad input for next iteration
			grad_input = &remembered_grad;

			// Remove the just used unfolded inner network, as
			// it is no longer necessary.
			forward_nets.pop_back();
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
		serialize(net.forward_nets, out);
		serialize(net.trained_params, out);
		serialize(net.remember_input, out);
		serialize(net.in_sample_aliaser, out);
		serialize(net.out_sample_aliaser, out);
		serialize(net.in_sample_size, out);
		serialize(net.out_sample_size, out);
		serialize(net.params_size, out);
		serialize(net.may_have_new_params, out);
		serialize(net.batch_is_full_sequence, out);
	}

	friend void deserialize(rnn_& net, std::istream& in)
	{
		std::string version;
		deserialize(version, in);
		if (version != "rnn_")
			throw serialization_error("Unexpected version '"+version+"' found while deserializing rnn_.");
		deserialize(net.forward_nets, in);
		deserialize(net.trained_params, in);
		deserialize(net.remember_input, in);
		deserialize(net.in_sample_aliaser, in);
		deserialize(net.out_sample_aliaser, in);
		deserialize(net.in_sample_size, in);
		deserialize(net.out_sample_size, in);
		deserialize(net.params_size, in);
		deserialize(net.may_have_new_params, in);
		deserialize(net.batch_is_full_sequence, in);
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

	auto& get_memory_layer(INTERNALS &internal)
	{
		return layer<tag_rnn_memory>(internal).subnet().layer_details();
	}

	std::vector<INTERNALS> forward_nets;

	dlib::resizable_tensor trained_params;

	/**
	 * The data remembered from previous runs.
	 */
	dlib::resizable_tensor remember_input;

	alias_tensor in_sample_aliaser;
	alias_tensor out_sample_aliaser;

	size_t in_sample_size;
	size_t out_sample_size;

	size_t params_size;

	bool may_have_new_params;
	bool batch_is_full_sequence;
};


// Attempt to implement the RNN MUT1, empirically found to be
// better than ordinary LSTM in "An Empirical Exploration of
// Recurrent Network Architectures" by Rafal Jozefowicz,
// Wojciech Zaremba and Ilya Sutskever, given in:
// http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
//
// tag1: Whr h + Br, used in r
// tag2: Whh (h * r) + Bh, where r = sig(Whr h + Wxr x + Br)
// tag3: z
// tag4: First term of output: tanh(Whh (r * h) + tanh(x) + Bh) * z
template <unsigned long num_outputs>
using inner_lstm_mut_ =
	add_prev4<mul_prev<tag_rnn_memory, one_minus<skip3< // ... + h * (1 - z)
	    tag4<mul_prev<tag3, htan<add_prev<tag2, htan<skip_rnn_input< // z * tanh(Whh (h * sig(...)) + tanh(x) + Bh), first term of output
		tag3<sig<fc<num_outputs, skip_rnn_input< // z
		    tag2<fc<num_outputs, mul_prev<tag_rnn_memory, sig<add_prev1<fc_no_bias<num_outputs, skip_rnn_input< // Whh (h * sig(Whr h + Wxr x + Br)) + Bh
			tag1<fc<num_outputs, // Whr h + Br, used in r
			    rnn_subnet_base
			>>
		    >>>>>>>
		>>>>
	    >>>>>>
	>>>>;


template <unsigned long num_outputs, typename SUBNET>
using lstm_mut = add_layer<rnn_<inner_lstm_mut_<num_outputs>, num_outputs>, SUBNET>;
