# Just for fun, how far can you push high-level memory efficiency when running transformer architectures? 

A good test of this is seeing how efficient you can make an LLM run on barebones hardware, and seeing how large you can scale up those LLMs before things break. 

The Raspberry Pi is a great platform for this! To emulate the results, get a RasPi with 4GB of RAM and put in 32GB flash storage. 


Largest model to date: 

32B Deepseek R1 

Most performant model: 

16B Deepseek R1 at 1 token per second (these are considered good numbers). 

This repo has the "base" example, which contains many practices that are carried forward into more complex cases. 

Looking to add more cases moving forward, including how to get some of the larger models running. 

To use, clone the repo, then run the download script to download the trial model, the quantize script to prep it, the fix_vocab script if needed, and the raspi_llm script to run it.
