import re

# Đọc file logs
content = """
2025-07-03 16:36:53.086 | INFO     | maas.ext.maas.scripts.optimizer:test:173 - maas/ext/maas/scripts/optimized/GSM8K/train\round_6\GSM8K_controller_sample.pth
2025-07-03 16:36:55.881 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: In an industrial research lab, a scienti... Error: None
2025-07-03 16:36:55.881 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: In an industrial research lab, a scienti...
2025-07-03 16:36:55.889 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: In an industrial research lab, a scienti...'. Execution time: 2.5229 seconds.
2025-07-03 16:36:55.897 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: A chemist performs two reactions:

React... Error: None
2025-07-03 16:36:55.905 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: A chemist performs two reactions:

React...
2025-07-03 16:36:55.905 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: A chemist performs two reactions:

React...'. Execution time: 2.5390 seconds.
2025-07-03 16:36:55.914 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Astronomers are observing a planet with ... Error: None
2025-07-03 16:36:55.914 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Astronomers are observing a planet with ...
2025-07-03 16:36:55.914 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Astronomers are observing a planet with ...'. Execution time: 2.5479 seconds.
2025-07-03 16:36:55.928 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: How many of the stars listed below would... Error: None
2025-07-03 16:36:55.929 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: How many of the stars listed below would...
2025-07-03 16:36:55.929 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: How many of the stars listed below would...'. Execution time: 2.5636 seconds.
2025-07-03 16:36:55.929 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Consider a rhombohedral crystal, with th... Error: None
2025-07-03 16:36:55.939 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Consider a rhombohedral crystal, with th...
2025-07-03 16:36:55.939 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider a rhombohedral crystal, with th...'. Execution time: 2.5737 seconds.
2025-07-03 16:36:55.943 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: You are studying a nuclear decay which c... Error: None
2025-07-03 16:36:55.947 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: You are studying a nuclear decay which c...
2025-07-03 16:36:55.947 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: You are studying a nuclear decay which c...'. Execution time: 2.5812 seconds.
2025-07-03 16:36:55.947 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Which of the following (effective) parti... Error: None
2025-07-03 16:36:55.954 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Which of the following (effective) parti...
2025-07-03 16:36:55.958 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Which of the following (effective) parti...'. Execution time: 2.5918 seconds.
2025-07-03 16:36:55.961 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Consider an aperture, which shapes like ... Error: None
2025-07-03 16:36:55.962 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Consider an aperture, which shapes like ...
2025-07-03 16:36:55.964 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider an aperture, which shapes like ...'. Execution time: 2.5979 seconds.
2025-07-03 16:36:55.967 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: An electron is in the spin state (3i, 4)... Error: None
2025-07-03 16:36:55.970 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: An electron is in the spin state (3i, 4)...
2025-07-03 16:36:55.970 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: An electron is in the spin state (3i, 4)...'. Execution time: 2.6048 seconds.
2025-07-03 16:36:55.974 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: The reaction of an electron pair donor, ... Error: None
2025-07-03 16:36:55.977 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: The reaction of an electron pair donor, ...
2025-07-03 16:36:55.978 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: The reaction of an electron pair donor, ...'. Execution time: 2.6123 seconds.
2025-07-03 16:36:55.980 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Substances 1-6 undergo an electrophilic ... Error: None
2025-07-03 16:36:55.985 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Substances 1-6 undergo an electrophilic ...
2025-07-03 16:36:55.987 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Substances 1-6 undergo an electrophilic ...'. Execution time: 2.6213 seconds.
2025-07-03 16:36:56.986 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cycl... Error: None
2025-07-03 16:36:56.987 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cycl...
2025-07-03 16:36:56.987 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cycl...'. Execution time: 0.9981 seconds.
2025-07-03 16:36:56.990 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Which sequence of reactions from the fol... Error: None
2025-07-03 16:36:56.990 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Which sequence of reactions from the fol...
2025-07-03 16:36:56.990 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Which sequence of reactions from the fol...'. Execution time: 1.0015 seconds.
2025-07-03 16:36:56.996 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: α-β unsaturated carbonyls have a much mo... Error: None
2025-07-03 16:36:56.996 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: α-β unsaturated carbonyls have a much mo...
2025-07-03 16:36:56.996 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: α-β unsaturated carbonyls have a much mo...'. Execution time: 1.0073 seconds.
2025-07-03 16:36:57.180 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: You identified a new quorum-sensing pept... Error: None
2025-07-03 16:36:57.180 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: You identified a new quorum-sensing pept...
2025-07-03 16:36:57.180 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: You identified a new quorum-sensing pept...'. Execution time: 0.1844 seconds.
2025-07-03 16:36:57.270 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Astronomers are studying two binary star... Error: None
2025-07-03 16:36:57.270 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Astronomers are studying two binary star...
2025-07-03 16:36:57.270 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Astronomers are studying two binary star...'. Execution time: 0.0824 seconds.
2025-07-03 16:37:04.222 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Which of the following issues are the mo... Error: None
2025-07-03 16:37:04.223 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Which of the following issues are the mo...
2025-07-03 16:37:04.234 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Which of the following issues are the mo...'. Execution time: 10.8679 seconds.
2025-07-03 16:37:05.740 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: If uncertainty in space of electron's lo... Error: None
2025-07-03 16:37:05.741 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: If uncertainty in space of electron's lo...
2025-07-03 16:37:05.751 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: If uncertainty in space of electron's lo...'. Execution time: 9.7622 seconds.
2025-07-03 16:37:05.893 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Imagine an operator $\vec{P}$ of a syste... Error: None
2025-07-03 16:37:05.893 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Imagine an operator $\vec{P}$ of a syste...
2025-07-03 16:37:05.893 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Imagine an operator $\vec{P}$ of a syste...'. Execution time: 0.1342 seconds.
2025-07-03 16:37:07.877 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Consider an isolated system of 13 identi... Error: None
2025-07-03 16:37:07.880 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Consider an isolated system of 13 identi...
2025-07-03 16:37:07.883 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider an isolated system of 13 identi...'. Execution time: 14.5171 seconds.
2025-07-03 16:37:08.870 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Consider the following metric:

ds^{2}=\... Error: None
2025-07-03 16:37:08.880 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Consider the following metric:

ds^{2}=\...
2025-07-03 16:37:08.880 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider the following metric:

ds^{2}=\...'. Execution time: 12.8917 seconds.
2025-07-03 16:37:09.857 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Two stars are being studied. It has been... Error: None
2025-07-03 16:37:09.863 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Two stars are being studied. It has been...
2025-07-03 16:37:09.865 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Two stars are being studied. It has been...'. Execution time: 13.8761 seconds.
2025-07-03 16:37:09.931 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: What is the index of hydrogen deficiency... Error: None
2025-07-03 16:37:09.933 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: What is the index of hydrogen deficiency...
2025-07-03 16:37:09.933 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: What is the index of hydrogen deficiency...'. Execution time: 12.9371 seconds.
2025-07-03 16:37:14.966 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Consider a system with Hamiltonian opera... Error: None
2025-07-03 16:37:14.966 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Consider a system with Hamiltonian opera...
2025-07-03 16:37:14.973 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider a system with Hamiltonian opera...'. Execution time: 21.6070 seconds.
2025-07-03 16:37:15.304 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Astronomers are interested in the lumino... Error: None
2025-07-03 16:37:15.306 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Astronomers are interested in the lumino...
2025-07-03 16:37:15.314 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Astronomers are interested in the lumino...'. Execution time: 21.9488 seconds.
2025-07-03 16:37:16.337 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Consider a 1-dimensional relativistic ha... Error: None
2025-07-03 16:37:16.366 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Consider a 1-dimensional relativistic ha...
2025-07-03 16:37:16.379 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider a 1-dimensional relativistic ha...'. Execution time: 23.0137 seconds.
2025-07-03 16:37:16.705 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Observations of a quasar across the elec... Error: None
2025-07-03 16:37:16.738 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Observations of a quasar across the elec...
2025-07-03 16:37:16.829 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Observations of a quasar across the elec...'. Execution time: 23.4637 seconds.
2025-07-03 16:37:17.435 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: The state of a system at time t is given... Error: None
2025-07-03 16:37:17.521 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: The state of a system at time t is given...
2025-07-03 16:37:17.607 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: The state of a system at time t is given...'. Execution time: 24.2414 seconds.
2025-07-03 16:37:20.503 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: Very large number of neutrinos produced ... Error: None
2025-07-03 16:37:20.669 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: Very large number of neutrinos produced ...
2025-07-03 16:37:20.886 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Very large number of neutrinos produced ...'. Execution time: 24.8973 seconds.
2025-07-03 16:37:35.664 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: There has been an outbreak of an viral i...'. Execution time: 42.2986 seconds.
2025-07-03 16:37:42.923 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: The Mott-Gurney equation describes the d...'. Execution time: 46.9340 seconds.
2025-07-03 16:37:51.966 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: In an inactive state, a transcription fa...'. Execution time: 58.5999 seconds.
2025-07-03 16:37:57.186 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: In the lab, a chemist discovers an unnam...'. Execution time: 61.1975 seconds.
2025-07-03 16:37:59.336 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Calculate the amount of non-Gaussianity(...'. Execution time: 65.9701 seconds.
2025-07-03 16:38:01.580 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Identify the possible product when (1S,4...'. Execution time: 68.2139 seconds.
2025-07-03 16:38:08.505 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Given the following Lagrangian

\mathcal...'. Execution time: 75.1391 seconds.
2025-07-03 16:38:14.772 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Compound X, which has the following IR a...'. Execution time: 81.4063 seconds.
2025-07-03 16:38:14.788 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Imagine an uncharged spherical conductor...'. Execution time: 78.7988 seconds.
2025-07-03 16:38:19.006 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider an oscillating charge distribut...'. Execution time: 85.6406 seconds.
2025-07-03 16:38:19.013 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: We would like to dissolve (at 25°С) 0.1 ...'. Execution time: 83.0247 seconds.
2025-07-03 16:38:19.015 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: The study of quantum mechanics deals wit...'. Execution time: 85.6491 seconds.
2025-07-03 16:38:19.022 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider a system of three spins S1, S2 ...'. Execution time: 85.6561 seconds.
2025-07-03 16:38:20.770 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: We have a solution containing Co(II) ion...'. Execution time: 87.4046 seconds.
2025-07-03 16:38:20.770 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: The state of a system at time t is given...'. Execution time: 76.5298 seconds.
2025-07-03 16:38:21.697 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: While designing a high-resolution transm... Error: None
2025-07-03 16:38:21.701 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: While designing a high-resolution transm...
2025-07-03 16:38:21.706 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: While designing a high-resolution transm...'. Execution time: 88.3407 seconds.
2025-07-03 16:38:37.573 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: An atomic nucleus of mass M is at rest w... Error: None
2025-07-03 16:38:37.578 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: An atomic nucleus of mass M is at rest w...
2025-07-03 16:38:37.585 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: An atomic nucleus of mass M is at rest w...'. Execution time: 100.3150 seconds.
2025-07-03 16:38:39.543 | ERROR    | maas.ext.maas.benchmark.gsm8k:_generate_output:75 - Error during graph execution for input: Question: In a specific region of the sky, astrono... Error: None
2025-07-03 16:38:39.552 | WARNING  | maas.ext.maas.benchmark.gsm8k:evaluate_problem:104 - Output from graph execution is empty or None for problem: Question: In a specific region of the sky, astrono...
2025-07-03 16:38:39.559 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: In a specific region of the sky, astrono...'. Execution time: 102.5625 seconds.
2025-07-03 16:39:17.695 | INFO     | maas.ext.maas.benchmark.gsm8k:evaluate_problem:129 - Problem evaluated: 'Question: Consider the extension of the Standard M...'. Execution time: 144.3296 seconds.
"""

# Bắt tất cả giá trị thời gian
times = re.findall(r"Execution time:\s*([\d.]+)\s*seconds", content)

print(times)

# Chuyển sang float và tính tổng
total_seconds = sum(float(t) for t in times) / len(times)

print(f"Tổng thời gian thực thi: {total_seconds:.4f} seconds")
