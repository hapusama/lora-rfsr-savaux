$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install -e ".[rfsr]"

python -m unittest `
  weak_decoder.os_lora.tests.test_rf_super_resolution_frontend `
  weak_decoder.os_lora.tests.test_litenap_error_modes -v
