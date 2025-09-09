#pragma once

#include "parameters.cuh"

// A derived Parameters class that provides the bool-ctor
// previously declared in main_nep::Parameters.
class NepParameters : public Parameters {
public:
//   explicit NepParameters(bool skip_nep_in);

  void load_from_nep_txt(const std::string& filename, std::vector<float>& elite);

};

