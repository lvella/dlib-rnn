#ifndef RNN_H
#define RNN_H

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
		auto& t1 = sub.get_output();
		auto& t2 = layer<tag>(sub).get_output();
		output.set_size(t1.num_samples(), t1.k(), t1.nr(), t1.nc());

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

	// TODO: if ever we don't need to unroll the DAG, use instead:
	// std::reference_wrapper<const tensor> output;
	dlib::resizable_tensor output;
	dlib::resizable_tensor gradient;
};

template <typename SUBNET>
using constant = add_layer<constant_, SUBNET>;

/* A fc_ layer specialization where bias is initialized with a value 2,
 * useful as input for forget layer in a LSTM, where this high value
 * would saturate sigmoid function to 1.
 */
template <unsigned long num_outputs_>
class fc_high_bias_:
	public fc_<num_outputs_, FC_HAS_BIAS>
{
	public:
	template <typename SUBNET>
	void setup (const SUBNET& sub)
	{
		fc_<num_outputs_, FC_HAS_BIAS>::setup(sub);
		this->get_biases() = 2.0f;
	}
};

template <unsigned long num_outputs, typename SUBNET>
using fc_high_bias = add_layer<fc_high_bias_<num_outputs>, SUBNET>;

enum split_side
{
	SPLIT_LEFT = 0,
	SPLIT_RIGHT = 1
};

template <split_side SIDE, unsigned COUNT = 0>
class split_
{
public:
	template <typename SUBNET>
	void setup (const SUBNET& sub)
	{
		auto &in = sub.get_output();
		if(COUNT == 0) {
			assert(in.k() % 2 == 0);
			out_k = in.k() / 2;
			in_offset = out_k * SIDE;
		} else {
			out_k = COUNT;
			if(SIDE == SPLIT_LEFT) {
				in_offset_k = 0;
			} else {
				in_offset_k = in.k() - out_k;
			}
		}
		out_sample_size = out_k * in.nr() * in.nc();
		in_sample_size = in.k() * in.nr() * in.nc();
		in_offset = in_offset_k * in.nr() * in.nc();
	}

	template <typename SUBNET>
	void forward(const SUBNET& sub, dlib::resizable_tensor& data_output)
	{
		auto &in = sub.get_output();

		data_output.set_size(in.num_samples(), out_k, in.nr(), in.nc());

		tt::copy_tensor(data_output, 0, in, in_offset_k, out_k);
	}

	template <typename SUBNET>
	void backward(
	    const tensor& gradient_input,
	    SUBNET& sub,
	    tensor& /* params_grad */)
	{
		auto &grad_out = sub.get_gradient_input();

		// TODO: find a way to do it using tensor tools...
		const float *in_data = gradient_input.host();
		float *out_data = grad_out.host();

		size_t num_samples = gradient_input.num_samples();

		for(size_t s = 0; s < num_samples; ++s) {
			const float *in = &in_data[s * out_sample_size];
			float *out = &out_data[s * in_sample_size];
			for(size_t i = 0; i < out_sample_size; ++i) {
				out[i + in_offset] += in[i];
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

	friend void serialize(const split_& net, std::ostream& out)
	{
		serialize("split_", out);
		serialize(int(SIDE), out);
		serialize(net.out_k, out);
		serialize(net.in_offset, out);
		serialize(net.in_offset_k, out);
		serialize(net.in_sample_size, out);
		serialize(net.out_sample_size, out);
	}

	friend void deserialize(split_& net, std::istream& in)
	{
		std::string version;
		deserialize(version, in);
		if (version != "split_")
			throw serialization_error("Unexpected version '"+version+"' found while deserializing split_.");
		int side = 0;
		deserialize(side, in);
		if (SIDE != split_side(side))
			throw serialization_error("Wrong side found while deserializing split_");
		deserialize(net.out_k, in);
		deserialize(net.in_offset, in);
		deserialize(net.in_offset_k, in);
		deserialize(net.in_sample_size, in);
		deserialize(net.out_sample_size, in);
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

	size_t out_k;
	size_t in_offset, in_offset_k;
	size_t in_sample_size, out_sample_size;

	dlib::resizable_tensor params; // unused
};

template <typename SUBNET>
using split_left = add_layer<split_<SPLIT_LEFT>, SUBNET>;

template <typename SUBNET>
using split_right = add_layer<split_<SPLIT_RIGHT>, SUBNET>;

template <unsigned COUNT, typename SUBNET>
using split_left_count = add_layer<split_<SPLIT_LEFT, COUNT>, SUBNET>;

template <unsigned COUNT, typename SUBNET>
using split_right_count = add_layer<split_<SPLIT_RIGHT, COUNT>, SUBNET>;

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
		mini_batch(50),
		out_sample_aliaser(mini_batch, mem_k, mem_nr, mem_nc)
	{
		remember_input.set_size(mini_batch, mem_k, mem_nr, mem_nc);

		dbg_count = 0;
		dbg_sum = 0;
		dbg_max = 0;
	}

	rnn_(const rnn_&) = default;

	template<typename F>
	void set_reseter(const F& func)
	{
		reserter = func;
	}

	void reset_sequence()
	{
		if(reserter)
			remember_input = 0.0f;
		else
			reserter(remember_input);
	}

	void set_mini_batch_size(size_t mini_batch_size)
	{
		mini_batch = mini_batch_size;
		in_sample_aliaser = alias_tensor(mini_batch,
			in_sample_aliaser.k(),
			in_sample_aliaser.nr(),
			in_sample_aliaser.nc());

		out_sample_aliaser = alias_tensor(mini_batch,
			remember_input.k(),
			remember_input.nr(),
			remember_input.nc());

		remember_input.set_size(mini_batch,
			remember_input.k(),
			remember_input.nr(),
			remember_input.nc());
	}

	void set_batch_is_full_sequence(bool is_full_sequence)
	{
		batch_is_full_sequence = is_full_sequence;
	}

	void set_for_run()
	{
		set_batch_is_full_sequence(false);
		set_mini_batch_size(1);
		reset_sequence();
	}

	template <typename SUBNET>
	void setup (const SUBNET& sub)
	{
		auto &in = sub.get_output();

		// Setup sequence params
		reset_sequence();

		in_sample_size = in.k() * in.nr() * in.nc();
		in_sample_aliaser = alias_tensor(mini_batch, in.k(), in.nr(), in.nc());

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

		assert(num_samples % mini_batch == 0);
		size_t seq_size = num_samples / mini_batch;

		data_output.set_size(num_samples,
			remember_input.k(),
			remember_input.nr(),
			remember_input.nc());

		forward_nets.resize(1);
		forward_nets.reserve(seq_size);

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
			auto sample_input = in_sample_aliaser(in, s * mini_batch * in_sample_size);

			// Copy the remembered data to the inner network's
			// special memory layer. Every network buit upon
			// rnn_subnet_base will have it.
			auto& mem_layer = get_memory_layer(forward_nets[s]);
			mem_layer.set_constant(*remembered);

			// Pass the input tensor through the internal subnetwork
			auto &sout = forward_nets[s].forward(sample_input);

			// Copy the single output to the assembled outputs
			{
				auto dest = out_sample_aliaser(data_output, s * mini_batch * out_sample_size);
				memcpy(dest, sout);
			}

			// Test end of loop
			if(++s >= seq_size) {
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

		resizable_tensor remembered_grad(mini_batch, gradient_input.k(), gradient_input.nr(), gradient_input.nc());
		assert(out_sample_size == gradient_input.k() * gradient_input.nr() * gradient_input.nc());

		{
			// Zeroes the params grad output before accumulating.
			params_grad_out = 0.0f;
		}

		// Get the loop counter, backwards because this is a BACKpropagation.
		size_t s = (gradient_input.num_samples() / mini_batch) - 1;

		// Get the gradient slice used in the first iteration
		auto in_grad_slice = out_sample_aliaser(gradient_input, s * mini_batch * out_sample_size);
		const tensor *grad_input = &in_grad_slice.get();
		for(;;) {
			// Retrieve the interation's network
			auto &fnet = forward_nets[s];

			// Get the input tensor, the same use in forward operation
			auto sample_input = in_sample_aliaser(in, s * mini_batch * in_sample_size);

			// Do the backpropagation in the inner network and get the output.
			fnet.back_propagate_error(sample_input, *grad_input);
			const tensor& inner_data_grad = fnet.get_final_data_gradient();

			// Assign iteration data grad output in the full data output.
			{
				auto dest = in_sample_aliaser(data_grad_out, s * mini_batch * in_sample_size);
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
			in_grad_slice = out_sample_aliaser(gradient_input, s * mini_batch * out_sample_size);

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

		// Hack that will work only when all layers in the network are RNN.
		// This step will normalize the gradient to a value proportional
		// to the number of samples, for RNN accumulated gradients doesn't
		// scale linearly with the number of samples, but instead it
		// scales exponentially.
		//params_grad_out *= 2.0 / (1.0 + in.num_samples());

		{
			size_t size = params_grad_out.size();
			float *h = params_grad_out.host();
			for(size_t i = 0; i < size; ++i)
			{
				if(h[i] > 5.0)
					h[i] = 5.0;
				else if (h[i] < -5.0)
					h[i] = -5.0;

				float v = std::fabs(h[i]);
				if(v > dbg_max) {
					dbg_max = v;
				}
				dbg_sum += v;
				if(++dbg_count == 64000000) {
					std::cout << "### mean: " << dbg_sum / dbg_count << " (" << (dbg_sum == 0.0 ? "zero" : "non-zero")
						<< "),   max: " << dbg_max << " (" << (dbg_max == 0.0 ? "zero" : "nonzero") << ')' << std::endl;
					dbg_sum = dbg_max = dbg_count = 0;
				}
			}
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

	std::function<void(tensor&)> reserter;

	size_t in_sample_size;
	size_t out_sample_size;

	size_t mini_batch;

	size_t params_size;

	alias_tensor in_sample_aliaser;
	alias_tensor out_sample_aliaser;

	bool may_have_new_params;
	bool batch_is_full_sequence;

	size_t dbg_count;
	long double dbg_sum;
	float dbg_max;
};

template <unsigned long num_outputs, typename INTERNALS, typename SUBNET>
using rnn = add_layer<rnn_<INTERNALS, num_outputs>, SUBNET>;

// Implementation the RNN's architectures given in "An Empirical
// Exploration of Recurrent Network Architectures" by Rafal
// Jozefowicz, Wojciech Zaremba and Ilya Sutskever, found in:
// http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
//
// Following is the formulation of MUT1, found in the paper to
// be better than ordinary LSTM.
//
// This architecture requires that the input to be the same size of
// the output, given that the vector tanh(x) which has the same size
// as input x must be added to vector Bh, which has the same size as
// output h.
//
// MUT1:
// z = sigm(Wxz x + Bz)
// r = sigm(Wxr x + Whr h + Br)
// h ← tanh(Whh (r * h) + tanh(x) + Bh) * z
//   + h * (1 - z)
//
// where:
// x: network input at that time
// h: network output remembered from previous evaluation
// o: current network output, will became h on next evaluation
//
// The following is the equivalent of MUT1 but reordered as
// implemented:
//
// o  = (t4 + h * (1 - t3))
// t4 = (t3 * tanh(t2 + tanh(x)))
// t3 = sigm(Wxz x + Bz)
// t2 = Whh (h * sigm(t1 + Wxr x)) + Bh
// t1 = Whr h + Br
//
template <unsigned long num_outputs>
using inner_lstm_mut1_ =
	add_prev4<mul_prev<tag_rnn_memory, one_minus<skip3<
	tag4<mul_prev<tag3, htan<add_prev<tag2, htan<skip_rnn_input<
	tag3<sig<fc<num_outputs, skip_rnn_input<
	tag2<fc<num_outputs, mul_prev<tag_rnn_memory, sig<add_prev1<fc_no_bias<num_outputs, skip_rnn_input<
	tag1<fc<num_outputs,
	rnn_subnet_base
	>>>>>>>>>>>>>>>>>>>>>>>;

template <unsigned long num_outputs, typename SUBNET>
using lstm_mut1 = rnn<num_outputs, inner_lstm_mut1_<num_outputs>, SUBNET>;


// MUT2 RNN architecture, as given in the paper.
//
// Due to term r * h, r is required to have the same size as h,
// and due to term x in r, r has the same size as x, since h is
// the output, this layout also requires x to be the size as output.
//
// MUT2:
// z = sigm(Wxz x + Whz h + Bz)
// r = sigm(x + Whr h + Br)
// h ← tanh(Whh (r * h) + Wxh x + Bh) * z
//   + h * (1 - z)
//
// Which translates to:
//
// o  = t1 + h * (1 - t2)
// t1 = t2 * tanh(t4 + Wxh x + Bh)
// t2 = sigm(t3 + Whz h + Bz)
// t3 = Wxz x
// t4 = Whh (h * sigm(x + Whr h + Br))
//
template <unsigned long num_outputs>
using inner_lstm_mut2_ =
	add_prev1<mul_prev<tag_rnn_memory, one_minus<skip2<
	tag1<mul_prev<tag2, htan<add_prev4<fc<num_outputs, skip_rnn_input<
	tag2<sig<add_prev3<fc<num_outputs, skip_rnn_memory<
	tag3<fc_no_bias<num_outputs, skip_rnn_input<
	tag4<fc_no_bias<num_outputs, mul_prev<tag_rnn_memory, sig<add_prev<tag_rnn_input, fc<num_outputs,
	rnn_subnet_base
	>>>>>>>>>>>>>>>>>>>>>>>>;

template <unsigned long num_outputs, typename SUBNET>
using lstm_mut2 = rnn<num_outputs, inner_lstm_mut2_<num_outputs>, SUBNET>;

// MUT3 RNN architecture, as given in the paper.
//
// MUT3:
// z = sigm(Wxz x + Whz tanh(h) + Bz)
// r = sigm(Wxr x + Whr h + Br)
// h ← tanh(Whh (r * h) + Wxh x + Bh) * z
//   + h * (1 - z)
//
// Which translates to:
//
// o  = t1 + h * (1 - t2)
// t1 = t2 * tanh(t4 + Wxh x + Bh)
// t2 = sigm(t3 + Wxz x + Bz)
// t3 = Whz tanh(h)
// t4 = Whh (h * sigm(t5 + Wxr x + Br))
// t5 = Wxr h
//
template <unsigned long num_outputs>
using inner_lstm_mut3_ =
	add_prev1<mul_prev<tag_rnn_memory, one_minus<skip2<
	tag1<mul_prev<tag2, htan<add_prev4<fc<num_outputs, skip_rnn_input<
	tag2<sig<add_prev3<fc<num_outputs, skip_rnn_input<
	tag3<fc_no_bias<num_outputs, htan<skip_rnn_memory<
	tag4<fc_no_bias<num_outputs, mul_prev<tag_rnn_memory, sig<add_prev5<fc<num_outputs, skip_rnn_input<
	tag5<fc_no_bias<num_outputs,
	rnn_subnet_base
	>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

template <unsigned long num_outputs, typename SUBNET>
using lstm_mut3 = rnn<num_outputs, inner_lstm_mut3_<num_outputs>, SUBNET>;

// Gate Recurrent Unit (GRU), as given in the paper.
//
// GRU:
// r = sigm(Wxr x + Whr h + Br)
// z = sigm(Wxz x + Whz h + Bz)
// g = tanh(Wxh x + Whh (r * h) + Bh)
// h ← z * h + (1 - z) * g
//
// Which is, equivalently, implemented as:
//
// o  = t1 + t2 * tanh(t5 + Wxh x + Bh)
// t1 = h * t3
// t2 = 1 - t3
// t3 = sigm(t4 + Wxz x + Bz)
// t4 = Whz h
// t5 = Whh (h * sigm(t6 + Wxr x + Br))
// t6 = Whr h
//
template <unsigned long num_outputs>
using inner_gru_ =
	add_prev1<mul_prev<tag2, htan<add_prev5<fc<num_outputs, skip_rnn_input<
	tag1<mul_prev<tag_rnn_memory, skip3<
	tag2<one_minus<
	tag3<sig<add_prev4<fc<num_outputs, skip_rnn_input<
	tag4<fc_no_bias<num_outputs, skip_rnn_memory<
	tag5<fc_no_bias<num_outputs, sig<add_prev6<fc<num_outputs, skip_rnn_input<
	tag6<fc_no_bias<num_outputs,
	rnn_subnet_base
	>>>>>>>>>>>>>>>>>>>>>>>>>>>;

template <unsigned long num_outputs, typename SUBNET>
using gru = rnn<num_outputs, inner_gru_<num_outputs>, SUBNET>;

// Needed for next network.
template <typename SUBNET> using tag11 = add_tag_layer<11, SUBNET>;

// Long Short-Term Memory (LSTM), as given in the paper.
//
// LSTM:
// i = tanh(Wxi x + Whi h + Bi)
// j = sigm(Wxj x + Whj h + Bj)
// f = sigm(Wxf x + Whf h + Bf)
// o = tanh(Wxo x + Who h + Bo)
// c ← c * f + i * j
// h ← tanh(c) * o
//
// which translates to:
//
// o   = t1, t11
// t11 = t7 * tanh(t1)
// t1  = t3 + t5 * sigm(t2 + Whj t9 + Bj)
// t2  = Wxj x
// t3  = t10 * sigm(t4 + Whf t9 + Bf)
// t4  = Wxf x
// t5  = tanh(t6 + Whi t9 + Bi)
// t6  = Wxi x
// t7  = tanh(t8 + Who t9 + Bo)
// t8  = Wxo x
// t9  = (c, h)[1]
// t10 = (c, h)[0]
//
template <unsigned long num_outputs>
using inner_lstm1_ =
	concat2<tag1, tag11,
	tag11<mul_prev<tag7, htan<skip1<
	tag1<add_prev3<mul_prev<tag5, sig<add_prev2<fc<num_outputs, skip9<
	tag2<fc_no_bias<num_outputs, skip_rnn_input<
	tag3<mul_prev<tag10, sig<add_prev4<fc_high_bias<num_outputs, skip9<
	tag4<fc_no_bias<num_outputs, skip_rnn_input<
	tag5<htan<add_prev6<fc<num_outputs, skip9<
	tag6<fc_no_bias<num_outputs, skip_rnn_input<
	tag7<htan<add_prev8<fc<num_outputs, skip9<
	tag8<fc_no_bias<num_outputs, skip_rnn_input<
	tag9<split_right<skip_rnn_memory<
	tag10<split_left<
	rnn_subnet_base
	>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

template <unsigned long num_outputs, typename SUBNET>
using lstm1 = split_right<rnn<2 * num_outputs, inner_lstm1_<num_outputs>, SUBNET>>;

// Basic Long Short-Term Memory (LSTM), as given in the blog post:
// https://colah.github.io/posts/2015-08-Understanding-LSTMs/
//
// It is defined as:
// i = tanh(Wxi x + Whi h + Bi)
// j = sigm(Wxj x + Whj h + Bj)
// f = sigm(Wxf x + Whf h + Bf)
// o = sigm(Wxo x + Who h + Bo)
// c ← c * f + i * j
// h ← tanh(c) * o
//
// which translates to:
//
// o   = t1, t11
// t11 = t7 * tanh(t1)
// t1  = t3 + t5 * sigm(t2 + Whj t9 + Bj)
// t2  = Wxj x
// t3  = t10 * sigm(t4 + Whf t9 + Bf)
// t4  = Wxf x
// t5  = tanh(t6 + Whi t9 + Bi)
// t6  = Wxi x
// t7  = sigm(t8 + Who t9 + Bo)
// t8  = Wxo x
// t9  = (c, h)[1]
// t10 = (c, h)[0]

template <unsigned long num_outputs>
using inner_lstm2_ =
	concat2<tag1, tag11,
	tag11<mul_prev<tag7, htan<skip1<
	tag1<add_prev3<mul_prev<tag5, sig<add_prev2<fc<num_outputs, skip9<
	tag2<fc_no_bias<num_outputs, skip_rnn_input<
	tag3<mul_prev<tag10, sig<add_prev4<fc_high_bias<num_outputs, skip9<
	tag4<fc_no_bias<num_outputs, skip_rnn_input<
	tag5<htan<add_prev6<fc<num_outputs, skip9<
	tag6<fc_no_bias<num_outputs, skip_rnn_input<
	tag7<sig<add_prev8<fc<num_outputs, skip9<
	tag8<fc_no_bias<num_outputs, skip_rnn_input<
	tag9<split_right<skip_rnn_memory<
	tag10<split_left<
	rnn_subnet_base
	>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

template <unsigned long num_outputs, typename SUBNET>
using lstm2 = split_right<rnn<2 * num_outputs, inner_lstm2_<num_outputs>, SUBNET>>;


// To be used in building the input, because rnn_ expect
// the mini_batches grouped by sequence position.
template <typename Iter>
class transpose_iterator
{
public:
	static_assert(
		std::is_base_of<
			std::random_access_iterator_tag,
			typename std::iterator_traits<Iter>::iterator_category
		>::value,
		"tranpose_iterator can only be used on a RandomAccessIterator"
	);

	typedef ptrdiff_t difference_type;
	typedef typename std::iterator_traits<Iter>::value_type value_type;
	typedef typename std::iterator_traits<Iter>::reference reference;
	typedef typename std::iterator_traits<Iter>::reference pointer;
	typedef std::random_access_iterator_tag iterator_category;

	transpose_iterator(Iter first, Iter last, size_t row_size):
		counter(0),
		ncols(row_size),
		first(first)
	{
		size_t total_size = std::distance(first, last);
		assert(total_size % row_size == 0);

		nrows = total_size / row_size;
	}

	transpose_iterator& operator++()
	{
		++counter;
		return *this;
	}

	transpose_iterator operator++(int)
	{
		transpose_iterator copy(*this);
		++counter;
		return copy;
	}

	transpose_iterator& operator--()
	{
		--counter;
		return *this;
	}

	transpose_iterator operator--(int)
	{
		transpose_iterator copy(*this);
		--counter;
		return copy;
	}

	friend bool operator<(const transpose_iterator& a, const transpose_iterator& b)
	{
		return a.counter < b.counter;
	}

	friend bool operator>(const transpose_iterator& a, const transpose_iterator& b)
	{
		return a.counter > b.counter;
	}

	friend bool operator<=(const transpose_iterator& a, const transpose_iterator& b)
	{
		return a.counter <= b.counter;
	}

	friend bool operator>=(const transpose_iterator& a, const transpose_iterator& b)
	{
		return a.counter >= b.counter;
	}

	transpose_iterator& operator+=(size_t s)
	{
		counter += s;
		return *this;
	}

	friend transpose_iterator operator+(const transpose_iterator& i, size_t s)
	{
		transpose_iterator copy(i);
		copy.counter += s;
		return copy;
	}

	friend transpose_iterator operator+(size_t s, const transpose_iterator& i)
	{
		return i + s;
	}

	transpose_iterator& operator-=(size_t s)
	{
		counter -= s;
		return *this;
	}

	friend transpose_iterator operator-(const transpose_iterator& i, size_t s)
	{
		transpose_iterator copy(i);
		copy.counter -= s;
		return copy;
	}

	friend ptrdiff_t operator-(const transpose_iterator& a, const transpose_iterator& b)
	{
		return ptrdiff_t(a.counter) - ptrdiff_t(b.counter);
	}

	reference operator[](size_t s) const
	{
		size_t input_row = (counter + s) % nrows;
		size_t input_col = (counter + s) / nrows;
		return first[(input_row * ncols) + input_col];
	}

	reference operator*() const
	{
		return (*this)[0];
	}

	pointer operator->() const
	{
		return &**this;
	}

private:
	size_t counter;
	size_t nrows, ncols;

	Iter first;
};

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

#endif
