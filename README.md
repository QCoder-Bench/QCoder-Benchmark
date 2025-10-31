# QCoder Benchmark

  QCoder Benchmark is a benchmark that evaluates performance by having a large language model (LLM) solve contest problems from the quantum programming competition “[QCoder](https://www.qcoder.jp/en).”

## How to Use

### 1, Setup
  If you want to use local LLM such as Llama, DeepSeek, etc., please set up your envirnment to use vllm. 
  When using an OpenAI model such as GPT-5, please enter your own API key in the `api_key` field of `codes->keys.json`.

### 2, Sign up for QCoder
  Press this [link](https://www.qcoder.jp/en/auth/signup) to sign up for QCoder. At present, this benchmark imposes a limit of *100 code validation attempts within a 24-hour period*. Please *DO* *NOT* create multiple accounts to circumvent this restriction. We continuously monitor usage, and your account may be suspended if any suspicious activity is detected.

