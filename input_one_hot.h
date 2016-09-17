#ifndef INPUT_ONE_HOT_H
#define INPUT_ONE_HOT_H

#include <dlib/dnn.h>

template <typename T, size_t NUM_CLASSES>
class input_one_hot
{
public:

	// sample_expansion_factor must be > 0
	const static unsigned int sample_expansion_factor = 1;
	typedef T input_type;

	template <typename forward_iterator>
	void to_tensor (
		forward_iterator ibegin,
		forward_iterator iend,
		dlib::resizable_tensor& data
	) const
	{
		data.set_size(std::distance(ibegin, iend), NUM_CLASSES, 1, 1);
		float *h = data.host_write_only();
		std::fill(h, h + data.size(), 0.0f);

		for(size_t i = 0; ibegin != iend; ++ibegin, ++i) {
			T val = *ibegin;
			assert(val < NUM_CLASSES);
			h[i * NUM_CLASSES + val] = 1.0f;
		}
	}

	friend std::ostream& operator<<(std::ostream& out, const input_one_hot& item)
	{
		/* TODO */
		return out;
	}

	friend void serialize(const input_one_hot& item, std::ostream& out)
	{ /* TODO */ }

	friend void deserialize(input_one_hot& item, std::istream& in)
	{/* TODO */}

};

#endif
