#include <cassert>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llama.cpp/llama.h"

namespace {

namespace py = pybind11;

class plamo_cpp_model {
public:
  static std::unique_ptr<plamo_cpp_model>
  load_from_file(std::string const &model_file_path, size_t n_threads) {

    llama_model_params model_params = llama_model_default_params();
    llama_model *model =
        llama_load_model_from_file(model_file_path.c_str(), model_params);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed = 1234;
    ctx_params.n_ctx = 2048; // TODO
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch =
        n_threads; // params.n_threads_batch == -1 ? params.n_threads :
                   // params.n_threads_batch;
    llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    return std::make_unique<plamo_cpp_model>(
        plamo_cpp_model(std::move(model), ctx));
  }

  py::array_t<float> calc_next_token_logits(py::array_t<int> const &input_ids,
                                            size_t vocab_size) {
    py::print("input_ids = ", input_ids);
    assert(input_ids.shape(0) == 1); // batch_size must be 1
    llama_batch batch = llama_batch_init(2048, 0); // TODO
    batch.n_tokens = input_ids.shape(1);
    for (size_t i = 0; i < batch.n_tokens; ++i) {
      batch.token[i] = *input_ids.data(0, i); // TODO
      batch.pos[i] = i;
      batch.seq_id[i] = 0;
      batch.logits[i] = false;
    }
    batch.logits[batch.n_tokens - 1] = true;
    //if (auto result = llama_decode(ctx_, batch); result != 0) {
    if (auto result = llama_decode(ctx_, batch); result < 0) {
      throw std::runtime_error("llama_decode failed " + std::to_string(result));
    }
    auto *logits_data = llama_get_logits_ith(ctx_, batch.n_tokens - 1);
    py::array_t<float> logits(
        std::vector<size_t>{static_cast<size_t>(input_ids.shape(0)), 1u,
                            vocab_size},
        logits_data);
    py::print("logits = ", logits);
    /*
    py::array_t<float> logits(std::vector<size_t>{
        static_cast<size_t>(input_ids.shape(0)), 1u, vocab_size});
    */
    return logits;
  }

private:
  plamo_cpp_model(llama_model *model, llama_context *ctx)
      : model_(model), ctx_(ctx) {}
  llama_model *model_;
  llama_context *ctx_;
};

} // namespace

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(infer, m) {
  m.doc() = "infer module";

  m.def("load_model_from_file", &plamo_cpp_model::load_from_file, "",
        py::arg("model_file_path"), py::arg("n_threads"));

  py::class_<plamo_cpp_model, std::unique_ptr<plamo_cpp_model>>(
      m, "plamo_cpp_model")
      //.def(py::init<>()) // use load_model_from_file
      .def("calc_next_token_logits", &plamo_cpp_model::calc_next_token_logits,
           py::arg("input_ids"), py::arg("vocab_size"));

  m.def("add", &add, "A function that adds two numbers");
}
